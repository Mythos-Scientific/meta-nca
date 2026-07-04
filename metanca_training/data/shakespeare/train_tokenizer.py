import argparse

import sentencepiece as spm

parser = argparse.ArgumentParser()
parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--model-prefix", type=str, default="shakespeare_10k_bpe")
args = parser.parse_args()

spm.SentencePieceTrainer.Train(
    input="input.txt",
    model_prefix=args.model_prefix,
    vocab_size=args.vocab_size,
    model_type="bpe",
    character_coverage=1.0,
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    byte_fallback=True,          # any OOV byte still encodes -> no <unk> data loss
    normalization_rule_name="identity",  # keep Shakespeare text verbatim
    remove_extra_whitespaces=False,
)
sp = spm.SentencePieceProcessor()
sp.Load(f"{args.model_prefix}.model")
print("PIECE_SIZE", sp.GetPieceSize())
print("unk/bos/eos/pad", sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id())
ids = sp.EncodeAsIds("First Citizen:\nBefore we proceed any further, hear me speak.")
print("sample ids (n=%d):" % len(ids), ids[:20])
print("roundtrip:", repr(sp.DecodeIds(ids)))
