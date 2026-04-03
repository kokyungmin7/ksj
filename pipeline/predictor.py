from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from PIL import Image

from pipeline.router import is_counting_question
from models.qwen import qwen_predict, qwen_predict_with_crop
from models.grounding_dino import (
    get_grounding_crop,
    grounding_count,
    pick_answer_by_count,
    extract_object_noun,
)


class Predictor:
    """Routes each sample through GroundingDINO crop → Qwen3-VL or GroundingDINO count.

    If cfg.dino.enabled is False, GroundingDINO is not used:
    - No crop is generated
    - All questions go directly to Qwen with the original image
    - trace will contain None for all detection-related fields
    """

    def __init__(
        self,
        qwen_model,
        qwen_processor,
        dino_model,        # GroundingDINO model (or None)
        dino_processor,    # GroundingDINO processor (or None)
        cfg: SimpleNamespace,
        reference_dir: str | None = None,  # unused with GroundingDINO, kept for CLI compat
    ) -> None:
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.cfg = cfg
        self.dino_enabled = getattr(cfg.dino, "enabled", True)

    def predict(self, row: dict) -> str:
        """Return predicted answer letter (a/b/c/d)."""
        answer, _ = self.predict_with_trace(row)
        return answer

    def predict_with_trace(self, row: dict) -> tuple[str, dict]:
        """Return (answer, trace) with all intermediate results.

        trace keys:
            route          str        "grounding_count" | "qwen_with_crop" | "qwen_only"
            bbox           tuple|None union bbox (x1,y1,x2,y2); None if disabled
            blobs_bbox     list       individual detection boxes
            attn_map       None       compat placeholder
            attn_map_full  None       compat placeholder
            crop           PIL.Image|None
            used_full      bool|None
            dino_count     int|None
            text_prompt    str|None   object noun used as GroundingDINO prompt
            raw_answer     str
        """
        if not self.dino_enabled:
            return self._predict_qwen_only(row)
        return self._predict_with_grounding(row)

    def _predict_qwen_only(self, row: dict) -> tuple[str, dict]:
        answer = qwen_predict(
            self.qwen_model,
            self.qwen_processor,
            row,
            self.cfg.data.image_root,
            self.cfg,
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

    def _predict_with_grounding(self, row: dict) -> tuple[str, dict]:
        image_path = os.path.join(self.cfg.data.image_root, str(row["path"]))
        image = Image.open(image_path).convert("RGB")

        # Extract object noun from question for GroundingDINO prompt
        question = str(row["question"])
        text_prompt = extract_object_noun(question)

        # Step 1: detect objects and build crop
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

        # Step 2: counting questions → GroundingDINO count
        if is_counting_question(question):
            count = grounding_count(
                self.dino_model, self.dino_processor, image, text_prompt, self.cfg
            )
            trace["dino_count"] = count
            answer = pick_answer_by_count(count, row)

            if answer is not None:
                trace["route"] = "grounding_count"
                trace["raw_answer"] = str(count)
                return answer, trace

        # Step 3: Qwen with original + crop
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            crop_path = tmp.name
            crop_result["crop"].save(crop_path, format="JPEG")

        try:
            answer = qwen_predict_with_crop(
                self.qwen_model,
                self.qwen_processor,
                row,
                self.cfg.data.image_root,
                crop_path,
                self.cfg,
            )
        finally:
            os.unlink(crop_path)

        trace["route"] = "qwen_with_crop"
        trace["raw_answer"] = answer
        return answer, trace
