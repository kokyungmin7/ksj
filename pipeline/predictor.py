from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from PIL import Image

from pipeline.router import is_counting_question
from models.qwen import qwen_predict, qwen_predict_with_crop, qwen_batch_predict
from models.grounding_dino import (
    get_grounding_crop,
    pick_answer_by_count,
    extract_object_noun,
)


class Predictor:
    """Routes each sample through GroundingDINO → count or Qwen.

    Counting flow:
        GroundingDINO detects individual objects (bbox)
        → bbox count → pick_answer_by_count
        → fallback to Qwen with crop if no numeric choices

    Non-counting flow:
        GroundingDINO crop → Qwen with crop

    If cfg.dino.enabled is False, skip all detection and use Qwen only.
    """

    def __init__(
        self,
        qwen_model,
        qwen_processor,
        dino_model,
        dino_processor,
        cfg: SimpleNamespace,
    ) -> None:
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.cfg = cfg
        self.dino_enabled = getattr(cfg.dino, "enabled", True)

    def predict(self, row: dict) -> str:
        answer, _ = self.predict_with_trace(row)
        return answer

    def predict_with_trace(self, row: dict) -> tuple[str, dict]:
        """Return (answer, trace).

        trace keys:
            route         "grounding_count" | "qwen_with_crop" | "qwen_only"
            bbox          (x1,y1,x2,y2) | None
            blobs_bbox    list of individual GroundingDINO boxes
            attn_map      None  (compat placeholder)
            attn_map_full None  (compat placeholder)
            crop          PIL.Image | None
            used_full     bool | None
            dino_count    int | None
            text_prompt   str | None
            raw_answer    str
        """
        if not self.dino_enabled:
            return self._predict_qwen_only(row)
        return self._predict_with_grounding(row)

    def _predict_qwen_only(self, row: dict) -> tuple[str, dict]:
        answer = qwen_predict(
            self.qwen_model, self.qwen_processor,
            row, self.cfg.data.image_root, self.cfg,
        )
        trace = {
            "route": "qwen_only",
            "bbox": None,
            "blobs_bbox": [],
            "attn_map": None,
            "attn_map_full": None,
            "crop": None,
            "used_full": None,
            "dino_count": None,
            "text_prompt": None,
            "raw_answer": answer,
        }
        return answer, trace

    def predict_batch_with_trace(self, rows: list[dict]) -> list[tuple[str, dict]]:
        """Batch prediction. Falls back to sequential when GroundingDINO is enabled."""
        if self.dino_enabled:
            return [self.predict_with_trace(row) for row in rows]

        answers = qwen_batch_predict(
            self.qwen_model, self.qwen_processor,
            rows, self.cfg.data.image_root, self.cfg,
        )

        results = []
        for answer in answers:
            trace = {
                "route": "qwen_only",
                "bbox": None,
                "blobs_bbox": [],
                "attn_map": None,
                "attn_map_full": None,
                "crop": None,
                "used_full": None,
                "dino_count": None,
                "text_prompt": None,
                "raw_answer": answer,
            }
            results.append((answer, trace))
        return results

    def _predict_with_grounding(self, row: dict) -> tuple[str, dict]:
        image_path = os.path.join(self.cfg.data.image_root, str(row["path"]))
        image = Image.open(image_path).convert("RGB")

        question = str(row["question"])
        text_prompt = extract_object_noun(question, row)

        crop_result = get_grounding_crop(
            self.dino_model, self.dino_processor, image, text_prompt, self.cfg
        )

        trace: dict = {
            "bbox": crop_result["bbox"],
            "blobs_bbox": crop_result["blobs_bbox"],
            "attn_map": None,
            "attn_map_full": None,
            "crop": crop_result["crop"],
            "used_full": crop_result["used_full"],
            "dino_count": None,
            "text_prompt": text_prompt,
            "route": None,
            "raw_answer": None,
        }

        if is_counting_question(question):
            count = len(crop_result["blobs_bbox"])
            trace["dino_count"] = count

            answer = pick_answer_by_count(count, row)
            if answer is not None:
                trace["route"] = "grounding_count"
                trace["raw_answer"] = str(count)
                return answer, trace

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            crop_path = tmp.name
            crop_result["crop"].save(crop_path, format="JPEG")

        try:
            answer = qwen_predict_with_crop(
                self.qwen_model, self.qwen_processor,
                row, self.cfg.data.image_root, crop_path, self.cfg,
            )
        finally:
            os.unlink(crop_path)

        trace["route"] = "qwen_with_crop"
        trace["raw_answer"] = answer
        return answer, trace
