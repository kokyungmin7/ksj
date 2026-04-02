from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

from PIL import Image

from pipeline.router import is_counting_question
from models.qwen import qwen_predict_with_crop
from models.dino import (
    get_attention_crop,
    dino_count,
    pick_answer_by_count,
    load_reference_paths,
)


class Predictor:
    """Routes each sample through DINOv3 crop → Qwen3-VL or DINOv3 count."""

    def __init__(
        self,
        qwen_model,
        qwen_processor,
        dino_model,
        dino_processor,
        cfg: SimpleNamespace,
        reference_dir: str | None = None,
    ) -> None:
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.dino_model = dino_model
        self.dino_processor = dino_processor
        self.cfg = cfg

        self.reference_paths: list[str] | None = None
        ref_dir = reference_dir or cfg.dino.reference_dir
        if ref_dir:
            self.reference_paths = load_reference_paths(ref_dir)

    def predict(self, row: dict) -> str:
        """Return predicted answer letter (a/b/c/d)."""
        answer, _ = self.predict_with_trace(row)
        return answer

    def predict_with_trace(self, row: dict) -> tuple[str, dict]:
        """Return (answer, trace) where trace contains all intermediate results.

        trace keys:
            route          str   "dino_count" | "qwen_with_crop"
            bbox           tuple (x1, y1, x2, y2) in original pixel coords
            attn_map       np.ndarray  raw attention map (num_h × num_w)
            attn_map_full  np.ndarray  upsampled to original image size
            crop           PIL.Image   the cropped ROI
            used_full      bool        True if crop fell back to full image
            dino_count     int | None  estimated count (counting questions only)
            raw_answer     str         raw generated text before extraction
        """
        image_path = os.path.join(self.cfg.data.image_root, str(row["path"]))
        image = Image.open(image_path).convert("RGB")

        # Step 1: always extract attention crop via DINOv3
        crop_result = get_attention_crop(
            self.dino_model, self.dino_processor, image, self.cfg
        )

        trace: dict = {
            "bbox": crop_result["bbox"],
            "attn_map": crop_result["attn_map"],
            "attn_map_full": crop_result["attn_map_full"],
            "crop": crop_result["crop"],
            "used_full": crop_result["used_full"],
            "dino_count": None,
            "route": None,
            "raw_answer": None,
        }

        # Step 2: route by question type
        question = str(row["question"])

        if is_counting_question(question):
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
            # Fallback: choices had no numbers → use Qwen with crop

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
