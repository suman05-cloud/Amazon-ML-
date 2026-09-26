import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Amazon ML: business entity resolution")
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile", help="Stream data sizes, country mix, missing fields and match counts")
    profile.add_argument("--data", default="data set/student_resource/dataset")
    profile.add_argument("--output", default="artifacts/data_profile.json")
    prepare = sub.add_parser("prepare", help="Sample references, retrieve against the ENTIRE training pool")
    prepare.add_argument("--data", default="data set/student_resource/dataset")
    prepare.add_argument("--work", default="artifacts/experiment_v2")
    prepare.add_argument("--sample-size", type=int, default=8000)
    prepare.add_argument("--seed", type=int, default=2027)
    prepare.add_argument("--block-cap", type=int, default=400)
    prepare.add_argument("--top-k", type=int, default=80)
    train = sub.add_parser("train")
    train.add_argument("--work", default="artifacts/experiment_v2")
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    train.add_argument("--rounds", type=int, default=800)
    train.add_argument("--no-transfer", action="store_true")
    index = sub.add_parser("build-index")
    index.add_argument("--data", default="data set/student_resource/dataset")
    index.add_argument("--index", default="artifacts/test.sqlite")
    index.add_argument("--split", choices=("train", "test"), default="test")
    infer = sub.add_parser("predict")
    infer.add_argument("--data", default="data set/student_resource/dataset")
    infer.add_argument("--index", default="artifacts/test.sqlite")
    infer.add_argument("--model", default="artifacts/experiment_v2/model")
    infer.add_argument("--output", default="output")
    infer.add_argument("--batch-size", type=int, default=500)
    infer.add_argument("--threads", type=int, default=4)
    infer.add_argument("--shard", type=int, default=0)
    infer.add_argument("--shards", type=int, default=1)
    infer.add_argument("--limit", type=int, help="Smoke test only; produces .preview.tsv files")
    infer.add_argument("--validate", action="store_true", help="Run strict full-output validation after unsharded complete inference")
    finish = sub.add_parser("finish", help="Run prediction workers, merge and validate automatically")
    finish.add_argument("--data", default="data set/student_resource/dataset")
    finish.add_argument("--index", default="artifacts/test.sqlite")
    finish.add_argument("--model", default="artifacts/experiment_v2/model")
    finish.add_argument("--output", default="output")
    finish.add_argument("--workers", type=int, default=4)
    finish.add_argument("--threads", type=int, default=2)
    finish.add_argument("--batch-size", type=int, default=500)
    merge = sub.add_parser("merge")
    merge.add_argument("--output", default="output")
    merge.add_argument("--shards", type=int, default=1)
    check = sub.add_parser("validate", help="Strict disk-backed ID, coverage and subset checks")
    check.add_argument("--data", default="data set/student_resource/dataset")
    check.add_argument("--output", default="output")
    package = sub.add_parser("package", help="Validate and create the required submission zip")
    package.add_argument("--data", default="data set/student_resource/dataset")
    package.add_argument("--output", default="output")
    package.add_argument("--destination", default="submission.zip")
    package.add_argument("--root", default=".")
    package.add_argument("--documentation", default="Documentation_template.md")
    document = sub.add_parser("document", help="Generate methodology from the measured experiment report")
    document.add_argument("--work", default="artifacts/experiment_v2")
    document.add_argument("--output", default="Documentation_template.md")
    document.add_argument("--team", default="[Your Team Name]")
    document.add_argument("--members", default="[List all team members]")
    args = parser.parse_args()
    if args.command == "profile":
        from .profile import profile
        profile(args.data, args.output)
    elif args.command == "prepare":
        if min(args.sample_size, args.block_cap, args.top_k) < 1:
            parser.error("sample size, block cap and top-k must be positive")
        from .train import prepare as run
        run(args.data, args.work, args.sample_size, args.seed, args.block_cap, args.top_k)
    elif args.command == "train":
        from .train import fit
        fit(args.work, args.threads, tuple(args.seeds), args.rounds, not args.no_transfer)
    elif args.command == "build-index":
        from .blocking import DiskIndex
        idx = DiskIndex(args.index, writable=True)
        try:
            idx.build([Path(args.data) / args.split / f"{args.split}_source{i}.tsv" for i in (2, 3)])
        finally:
            idx.close()
    elif args.command == "predict":
        if args.validate and (args.limit is not None or args.shards != 1):
            parser.error("--validate requires a full, unsharded prediction run")
        from .predict import predict
        predict(args.data, args.index, args.model, args.output, args.batch_size, args.shard, args.shards, args.limit, args.threads)
        if args.validate:
            from .validate import validate
            validate(args.data, args.output)
    elif args.command == "merge":
        from .predict import merge as run
        run(args.output, args.shards)
    elif args.command == "finish":
        from .runner import finish
        finish(args.data, args.index, args.model, args.output, args.workers, args.threads, args.batch_size)
    elif args.command == "validate":
        from .validate import validate
        validate(args.data, args.output)
    elif args.command == "package":
        from .package import package
        package(args.data, args.output, args.destination, args.root, args.documentation)
    elif args.command == "document":
        from .documentation import document
        document(args.work, args.output, args.team, args.members)
