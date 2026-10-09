import importlib.metadata as metadata


class QwenBaseline:
    def __init__(self, config: dict, offline: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. Run this on the Linux H200 server.")
        if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This experiment requires exactly one visible NVIDIA H200 GPU")
        if config["dtype"] != "bfloat16" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 is unavailable on this GPU")
        self.torch = torch
        self.config = config
        requested_attention = config["attn_implementation"]
        if requested_attention == "auto":
            try:
                import flash_attn  # noqa: F401
                requested_attention = "flash_attention_2"
            except ImportError:
                requested_attention = "sdpa"
        if requested_attention not in {"flash_attention_2", "sdpa"}:
            raise ValueError("Attention implementation must be auto, flash_attention_2 or sdpa")
        args = dict(torch_dtype=torch.bfloat16, attn_implementation=requested_attention,
                    local_files_only=offline, revision=config["model_revision"])
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model_id"], **args).to("cuda").eval()
            self.attention = requested_attention
        except (ImportError, ValueError, RuntimeError):
            if config["attn_implementation"] != "auto" or requested_attention != "flash_attention_2":
                raise
            args["attn_implementation"] = "sdpa"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model_id"], **args).to("cuda").eval()
            self.attention = "sdpa"
        self.processor = AutoProcessor.from_pretrained(
            config["model_id"], revision=config["model_revision"],
            max_pixels=int(config["max_pixels"]), local_files_only=offline,
        )
        self.model_revision = getattr(self.model.config, "_commit_hash", None) or config["model_revision"]

    def metadata(self) -> dict:
        torch = self.torch
        gpu = torch.cuda.get_device_properties(0)
        return dict(torch_version=torch.__version__, transformers_version=metadata.version("transformers"),
                    qwen_vl_utils_version=metadata.version("qwen-vl-utils"),
                    cuda_version=torch.version.cuda, gpu_name=gpu.name, gpu_memory_bytes=gpu.total_memory,
                    attention_implementation=self.attention, dtype="bfloat16", model_revision=self.model_revision)

    def predict(self, question: str, image_path: str) -> tuple[str, float, float]:
        import time
        from qwen_vl_utils import process_vision_info
        from .prompting import make_messages

        start = time.perf_counter()
        messages = make_messages(question, image_path)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
        self.torch.cuda.synchronize()
        preprocess_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=int(self.config["max_new_tokens"]))
        self.torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - start
        generated = output[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return raw, preprocess_seconds, generation_seconds
