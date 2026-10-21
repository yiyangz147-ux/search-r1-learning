#!/usr/bin/env python3
"""Download public Search-R1 data and export its nested Parquet schema.

QA source: RUC-NLPIR/FlashRAG_datasets. The prompt follows the user-provided
Search-R1 README and upstream scripts/data_process/nq_search.py (Apache-2.0).
This is a standalone data utility; it does not require verl, torch, or a GPU.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
QA_REPO = "RUC-NLPIR/FlashRAG_datasets"
EVAL_SPLITS = {"nq": "test", "hotpotqa": "dev"}
PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {question}\n"
)


def curl_json(url):
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
         "--connect-timeout", "30", "--max-time", "120", url],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"无法读取数据源元信息: {url}\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def repository(endpoint, repo):
    info = curl_json(f"{endpoint}/api/datasets/{repo}/revision/main")
    revision = info["sha"]
    files = curl_json(f"{endpoint}/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000")
    return revision, {item["path"]: item for item in files if item["type"] == "file"}


def fingerprints(path):
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    return sha.hexdigest(), blob.hexdigest()


def verify_download(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        raise ValueError(f"文件大小不符: {path}")
    sha, blob = fingerprints(path)
    expected = metadata.get("lfs", {}).get("oid")
    if expected and sha != expected:
        raise ValueError(f"SHA256 校验失败: {path}")
    if not expected and blob != metadata["oid"]:
        raise ValueError(f"Git blob 校验失败: {path}")
    return sha


def download(endpoint, repo, revision, name, destination, metadata):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}"
    if destination.exists():
        digest = verify_download(destination, metadata)
        print(f"已校验，跳过下载: {destination.relative_to(ROOT) if destination.is_relative_to(ROOT) else destination}", flush=True)
    else:
        partial = destination.with_name(destination.name + ".partial")
        print(f"下载 {repo}/{name} ({metadata['size'] / 1_000_000:.1f} MB)", flush=True)
        if not partial.exists() or partial.stat().st_size < metadata["size"]:
            subprocess.run(
                ["curl", "--fail", "--location", "--show-error", "--progress-bar",
                 "--retry", "3", "--retry-all-errors", "--connect-timeout", "30",
                 "--continue-at", "-", "--output", str(partial), url], check=True,
            )
        digest = verify_download(partial, metadata)
        partial.replace(destination)
    return {"repository": repo, "revision": revision, "file": name,
            "url": url, "bytes": metadata["size"], "sha256": digest}


def arrow_schema():
    try:
        import pyarrow as pa
    except ImportError:
        raise RuntimeError("请先运行 python -m pip install -r requirements-data.txt") from None
    return pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([
            ("style", pa.string()),
            ("ground_truth", pa.struct([("target", pa.list_(pa.string()))])),
        ])),
        ("extra_info", pa.struct([
            ("split", pa.string()), ("index", pa.int64()),
            ("source_split", pa.string()), ("source_id", pa.string()),
        ])),
    ])


def sample(example, source, split, source_split, index):
    question = example.get("question")
    answers = example.get("golden_answers")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}/{source_split} 第 {index} 条问题为空")
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError(f"{source}/{source_split} 第 {index} 条答案格式错误")
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": PROMPT.format(question=question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": {"split": split, "index": index, "source_split": source_split,
                       "source_id": str(example.get("id", index))},
    }


def convert(input_file, output_file, source, split, source_split):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial = output_file.with_suffix(".parquet.partial")
    preview = []
    count = 0
    with pq.ParquetWriter(partial, arrow_schema(), compression="snappy") as writer:
        batch = []
        with input_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = sample(json.loads(line), source, split, source_split, count)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{input_file}:{line_number}: {error}") from error
                if len(preview) < 3:
                    preview.append(row)
                batch.append(row)
                count += 1
                if len(batch) == 1024:
                    writer.write_table(pa.Table.from_pylist(batch, schema=arrow_schema()))
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=arrow_schema()))
    if count == 0 or pq.ParquetFile(partial).metadata.num_rows != count:
        raise ValueError(f"输出为空或行数不符: {partial}")
    partial.replace(output_file)
    output_file.with_suffix(".sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in preview), encoding="utf-8",
    )
    print(f"生成 {output_file}: {count:,} 条", flush=True)
    return count


def merge(files, destination):
    import pyarrow.parquet as pq
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".parquet.partial")
    count = 0
    with pq.ParquetWriter(partial, arrow_schema(), compression="snappy") as writer:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                writer.write_batch(batch)
                count += batch.num_rows
    if pq.ParquetFile(partial).metadata.num_rows != count:
        raise ValueError("合并后的 Parquet 行数不符")
    partial.replace(destination)
    print(f"生成合并文件 {destination}: {count:,} 条", flush=True)
    return count


def save_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def prepare_qa(args):
    arrow_schema()
    revision, remote_files = repository(args.endpoint, QA_REPO)
    manifest = {"repository": QA_REPO, "revision": revision, "downloads": [], "datasets": {}}
    for source in args.datasets:
        counts = {}
        for split, source_split in [("train", "train"), ("test", EVAL_SPLITS[source])]:
            name = f"{source}/{source_split}.jsonl"
            raw = args.data_dir / "raw" / name
            manifest["downloads"].append(download(args.endpoint, QA_REPO, revision, name, raw, remote_files[name]))
            output = args.data_dir / f"{source}_search" / f"{split}.parquet"
            counts[split] = {"rows": convert(raw, output, source, split, source_split),
                             "source_split": source_split, "path": str(output),
                             "sha256": fingerprints(output)[0]}
        manifest["datasets"][source] = counts
    if len(args.datasets) > 1:
        combined = args.data_dir / ("_".join(args.datasets) + "_search")
        manifest["merged"] = {}
        for split in ("train", "test"):
            output = combined / f"{split}.parquet"
            files = [args.data_dir / f"{source}_search" / f"{split}.parquet" for source in args.datasets]
            manifest["merged"][split] = {"rows": merge(files, output), "path": str(output),
                                         "sha256": fingerprints(output)[0]}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.data_dir / "qa_manifest.json", manifest)


def iter_corpus_lines(path):
    """Read either gzip JSONL or the upstream gzip TAR containing one JSONL file.

    The upstream wiki-18.jsonl.gz is a gzip-compressed TAR despite its filename.
    Read its payload as a stream, never extracting archive paths onto disk.
    """
    try:
        with gzip.open(path, "rb") as source:
            header = source.read(512)
            source.seek(0)
            try:
                tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
            except tarfile.HeaderError:
                yield from source
                return
            found = False
            with tarfile.open(fileobj=source, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if found or not member.name.endswith(".jsonl"):
                        raise ValueError("语料归档应只包含一个 JSONL 普通文件")
                    found = True
                    with archive.extractfile(member) as payload:
                        yield from payload
                if not found:
                    raise ValueError("语料归档没有 JSONL 普通文件")
    except (tarfile.TarError, EOFError) as error:
        raise ValueError(f"语料压缩包损坏或 TAR 格式无效: {type(error).__name__}") from None


def prepare_retrieval(args):
    corpus_dir = args.data_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    assets = [("PeterJinGo/wiki-18-e5-index", ["part_aa", "part_ab"]),
              ("PeterJinGo/wiki-18-corpus", ["wiki-18.jsonl.gz"])]
    manifest = {"retriever_model": "intfloat/e5-base-v2", "downloads": []}
    resolved = [(repo, names, *repository(args.endpoint, repo)) for repo, names in assets]
    total = sum(files[name]["size"] for _, names, _, files in resolved for name in names)
    print(f"完整检索资源需要下载 {total / 1e9:.2f} GB；合并索引时还需要约 64.6 GB 临时空间。", flush=True)
    if shutil.disk_usage(corpus_dir).free < total + 65_000_000_000 + 20_000_000_000:
        raise RuntimeError("磁盘空间不足：请预留下载、合并索引及解压语料所需的约 150 GB 空间。")
    for repo, names, revision, remote_files in resolved:
        for name in names:
            manifest["downloads"].append(download(args.endpoint, repo, revision, name, corpus_dir / name, remote_files[name]))
    index = corpus_dir / "e5_Flat.index"
    partial = index.with_suffix(".index.partial")
    index_sha = hashlib.sha256()
    with partial.open("wb") as output:
        for name in ("part_aa", "part_ab"):
            with (corpus_dir / name).open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    index_sha.update(block)
    expected = sum((corpus_dir / name).stat().st_size for name in ("part_aa", "part_ab"))
    if partial.stat().st_size != expected:
        raise ValueError("合并索引的大小不符")
    partial.replace(index)
    corpus = corpus_dir / "wiki-18.jsonl"
    partial = corpus.with_suffix(".jsonl.partial")
    print("解压 Wikipedia 语料并检查每行 id/contents 字段……", flush=True)
    corpus_sha = hashlib.sha256()
    count = 0
    with partial.open("wb") as output:
        for line in iter_corpus_lines(corpus_dir / "wiki-18.jsonl.gz"):
            item = json.loads(line)
            if not isinstance(item.get("contents"), str) or "id" not in item:
                raise ValueError(f"语料第 {count + 1} 行缺少有效 id/contents")
            output.write(line)
            corpus_sha.update(line)
            count += 1
    partial.replace(corpus)
    manifest["index"] = {"path": str(index), "bytes": index.stat().st_size, "sha256": index_sha.hexdigest()}
    manifest["corpus"] = {"path": str(corpus), "bytes": corpus.stat().st_size,
                          "passages": count, "sha256": corpus_sha.hexdigest()}
    save_json(corpus_dir / "retrieval_manifest.json", manifest)
    print(f"检索数据准备完成：{count:,} 个段落。原始压缩包及索引分片保留，可在确认后自行删除。", flush=True)


def check(args):
    import pyarrow.parquet as pq
    manifest_path = args.data_dir / "qa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source, splits in manifest["datasets"].items():
        for split, metadata in splits.items():
            path = Path(metadata["path"])
            parquet = pq.ParquetFile(path)
            if not parquet.schema_arrow.equals(arrow_schema(), check_metadata=False):
                raise ValueError(f"Parquet schema 不符: {path}")
            if parquet.metadata.num_rows != metadata["rows"] or fingerprints(path)[0] != metadata["sha256"]:
                raise ValueError(f"Parquet 行数或 SHA256 不符: {path}")
            count = 0
            for batch in parquet.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    if row["data_source"] != source or row["extra_info"]["split"] != split:
                        raise ValueError(f"样本来源/划分不符: {path}, 第 {count} 条")
                    if not row["reward_model"]["ground_truth"]["target"] or row["prompt"][0]["role"] != "user":
                        raise ValueError(f"样本格式不符: {path}, 第 {count} 条")
                    count += 1
            print(f"通过: {source}/{split}, {count:,} 条", flush=True)
    for split, metadata in manifest.get("merged", {}).items():
        path = Path(metadata["path"])
        expected = sum(splits[split]["rows"] for splits in manifest["datasets"].values())
        if pq.ParquetFile(path).metadata.num_rows != expected or fingerprints(path)[0] != metadata["sha256"]:
            raise ValueError(f"合并数据行数或 SHA256 不符: {path}")
        print(f"通过: merged/{split}, {expected:,} 条", flush=True)
    print("问答数据校验完成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="获取并处理 Search-R1 官方英文数据")
    parser.add_argument("command", choices=["qa", "retrieval", "check"])
    parser.add_argument("--datasets", nargs="+", choices=sorted(EVAL_SPLITS), default=["nq", "hotpotqa"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    args = parser.parse_args()
    args.endpoint = args.endpoint.rstrip("/")
    args.data_dir = args.data_dir.expanduser().resolve()
    args.datasets = list(dict.fromkeys(args.datasets))
    try:
        {"qa": prepare_qa, "retrieval": prepare_retrieval, "check": check}[args.command](args)
    except KeyboardInterrupt:
        print("\n已中断；下载临时文件保留，重新运行同一命令可续传。", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"失败: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
