import shutil

from mtb import REPO_ROOT
from mtb.visualization.create_index import create_index


def test_create_index(tmp_path):
    # Create a fake benchmark directory
    measurement_dir = tmp_path / "measurements"
    llm_benchmarks_dir = measurement_dir / "llm_benchmarks" / "Apple M4"
    layer_benchmarks_dir = measurement_dir / "layer_benchmarks" / "Apple M4"

    llm_benchmarks_dir.mkdir(parents=True, exist_ok=True)
    layer_benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # Create a "measurement" html file in each folder
    llm_benchmark_file = llm_benchmarks_dir / "llm_benchmark.html"
    layer_benchmark_file = layer_benchmarks_dir / "layer_benchmark.html"

    with llm_benchmark_file.open("w") as f:
        f.write("<html><body>LLM Benchmark</body></html>")

    with layer_benchmark_file.open("w") as f:
        f.write("<html><body>Layer Benchmark</body></html>")

    # Copy index template
    index_template_path = REPO_ROOT / "visualizations" / "index_template.html"
    shutil.copy(index_template_path, measurement_dir / "index_template.html")

    # Create the index
    index_path = create_index(measurement_dir)
    assert index_path.exists()
    assert index_path.name == "index.html"
