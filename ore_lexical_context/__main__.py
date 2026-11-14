"""Run a small lexical-context demonstration and optionally save JSON."""
from __future__ import annotations
import argparse
import json
from . import G2PRule, LexicalContextProcessor, LexicalContextRequest, LexicalEntry, LexicalMeaning, MorphemeEntry

def build_demo() -> dict[str, object]:
    lexicon = (
        LexicalEntry("cat","cat",frequency=30,phonemes=("K","AE","T"),
                     meanings=(LexicalMeaning("feline",{"ANIMAL":1.0,"FELINE":1.0}),),
                     verified_by="synthetic-demo",source_reference="synthetic://lexicon/cat"),
        LexicalEntry("bank","bank",frequency=22,meanings=(
            LexicalMeaning("river-bank",{"RIVER":1.0,"LAND":0.7}),
            LexicalMeaning("financial-bank",{"MONEY":1.0,"INSTITUTION":0.7}),
        ),verified_by="synthetic-demo",source_reference="synthetic://lexicon/bank"),
    )
    morphemes=(MorphemeEntry("plural-s","s","suffix",{"PLURAL":1.0},productivity=0.9,
                             verified_by="synthetic-demo",source_reference="synthetic://morpheme/s"),)
    rules=(G2PRule("c",("K",),0.95),G2PRule("a",("AE",),0.9),
           G2PRule("t",("T",),0.95),G2PRule("s",("S",),0.9))
    processor=LexicalContextProcessor(lexicon,morphemes=morphemes,g2p_rules=rules,
                                      bigram_counts={"the":{"cat":9},"cat":{"runs":5}})
    return processor.process(LexicalContextRequest("lexical-context-demo",text="the cats bank",
        context_features={"ANIMAL":0.7},source_reference="synthetic://ore-lexical-context-demo")).to_dict()

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",help="optional JSON output path")
    args=parser.parse_args()
    encoded=json.dumps(build_demo(),ensure_ascii=False,indent=2)
    if args.output:
        with open(args.output,"w",encoding="utf-8") as handle: handle.write(encoded+"\n")
    print(encoded)

if __name__=="__main__": main()
