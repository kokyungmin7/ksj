"""train.csv의 질문·선택지를 Darong/BlueT로 한→영 번역하여 train_en.csv로 저장."""

import pandas as pd
from transformers import pipeline, T5TokenizerFast
from tqdm import tqdm

CSV_IN = "train.csv"
CSV_OUT = "train_en.csv"
TRANSLATE_COLS = ["question", "a", "b", "c", "d"]
BATCH_SIZE = 64
PREFIX = "K2E: "


def build_translator():
    tokenizer = T5TokenizerFast.from_pretrained("paust/pko-t5-base")
    return pipeline(
        "translation",
        model="Darong/BlueT",
        tokenizer=tokenizer,
        max_length=255,
        device=0,
    )


def translate_column(translator, texts: list[str]) -> list[str]:
    prefixed = [PREFIX + t for t in texts]
    results: list[str] = []
    for i in tqdm(range(0, len(prefixed), BATCH_SIZE), desc="translating"):
        batch = prefixed[i : i + BATCH_SIZE]
        out = translator(batch, batch_size=BATCH_SIZE)
        results.extend(item["translation_text"] for item in out)
    return results


def main():
    df = pd.read_csv(CSV_IN)
    translator = build_translator()

    for col in TRANSLATE_COLS:
        print(f"\n>>> Translating column: {col}")
        df[col] = translate_column(translator, df[col].astype(str).tolist())

    df.to_csv(CSV_OUT, index=False)
    print(f"\nDone — saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
