from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from PIL import Image

from pipeline.router import is_counting_question
from models.qwen import qwen_predict, qwen_predict_with_crop
from models.dino import (
    get_attention_crop,
    dino_count,
    pick_answer_by_count,
    load_reference_paths,
)


class Predictor:
    """Routes each sample through DINOv3 crop → Qwen3-VL or DINOv3 count.

    If cfg.dino.enabled is False, DINOv3 is not used at all:
    - No crop is generated
    - All questions go directly to Qwen with the original image
    - trace will contain None for all DINOv3-related fields
    """

    def __init__(
        self,
        qwen_model,
        qwen_processor,
        dino_model,           # None when cfg.dino.enabled is False
        dino_processor,       # None when cfg.dino.enabled is False
        cfg: SimpleNamespace,
        reference_dir: str | None = None,
    ) -> None:
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.cfg = cfg
        self.dino_enabled = getattr(cfg.dino, "enabled", True)

        self.reference_paths: list[str] | None = None
        if self.dino_enabled:
            ref_dir = reference_dir or cfg.dino.reference_dir
            if ref_dir:
                self.reference_paths = load_reference_paths(ref_dir)

    def predict(self, row: dict) -> str:
        """Return predicted answer letter (a/b/c/d)."""
        answer, _ = self.predict_with_trace(row)
        return answer

    def predict_with_trace(self, row: dict) -> tuple[str, dict]:
        """Return (answer, trace) with all intermediate results.

        trace keys:
            route          str        "dino_count" | "qwen_with_crop" | "qwen_only"
            bbox           tuple|None (x1, y1, x2, y2); None if dino disabled
            attn_map       ndarray|None
            attn_map_full  ndarray|None
            crop           PIL.Image|None
            used_full      bool|None
            dino_count     int|None
            raw_answer     str
        """
        if not self.dino_enabled:
            return self._predict_qwen_only(row)
        return self._predict_with_dino(row)

    def _predict_qwen_only(self, row: dict) -> tuple[str, dict]:
        """Qwen inference on original image only (DINOv3 disabled)."""
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
            "raw_answer": answer,
        }
        return answer, trace

    def _predict_with_dino(self, row: dict) -> tuple[str, dict]:
        """DINOv3 crop + routing logic."""
        image_path = os.path.join(self.cfg.data.image_root, str(row["path"]))
        image = Image.open(image_path).convert("RGB")

        # Step 1: extract attention crop
        crop_result = get_attention_crop(
            self.dino_model, self.dino_processor, image, self.cfg
        )

        trace: dict = {
            "bbox": crop_result["bbox"],
            "blobs_bbox": crop_result["blobs_bbox"],
            "attn_map": crop_result["attn_map"],
            "attn_map_full": crop_result["attn_map_full"],
            "crop": crop_result["crop"],
            "used_full": crop_result["used_full"],
            "dino_count": None,
            "route": None,
            "raw_answer": None,
        }

        # Step 2: counting questions → DINOv3 count
        if is_counting_question(str(row["question"])):
            count = dino_count(
                self.dino_model,
                self.dino_processor,
                image_path,
                self.cfg,
                reference_paths=self.reference_paths,
            )
            trace["dino_count"] = count
            answer = pick_answer_by_count(count, row)

            if answer is not None:
                trace["route"] = "dino_count"
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
