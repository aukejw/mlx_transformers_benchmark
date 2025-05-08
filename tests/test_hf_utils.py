from unittest.mock import patch

from mtb.hf_utils import get_hf_home, set_hf_home, verbose_download_model


def test_set_home():
    # Set a new HF_HOME path
    new_path = "/tmp/huggingface"
    set_hf_home(new_path)
    assert get_hf_home() == new_path

    set_hf_home()
    assert get_hf_home() == "~/.cache/huggingface"


def test_verbose_download_model():
    model_id = "model_id"
    hf_home = get_hf_home()

    # Test when model is already downloaded
    with patch("mtb.hf_utils.snapshot_download") as mock_snapshot:
        mock_snapshot.return_value = "/cache/models/model_id"
        result = verbose_download_model(model_id)

        mock_snapshot.assert_called_once_with(
            model_id,
            local_files_only=True,
            cache_dir=hf_home,
        )
        assert result == "/cache/models/model_id"

    # Test when model needs to be downloaded
    with patch("mtb.hf_utils.snapshot_download") as mock_snapshot:

        def snapshot_download_function(
            model_id,
            local_files_only,
            cache_dir,
        ):
            if local_files_only:
                raise Exception("Model not found locally")
            return "/cache/models/model_id"

        mock_snapshot.side_effect = snapshot_download_function
        result = verbose_download_model(model_id)

        # Assert snapshot_download was called twice with correct parameters
        assert mock_snapshot.call_count == 2
        mock_snapshot.assert_any_call(
            model_id,
            local_files_only=True,
            cache_dir=hf_home,
        )
        mock_snapshot.assert_any_call(
            model_id,
            local_files_only=False,
            cache_dir=hf_home,
        )
        assert result == "/cache/models/model_id"
