"""Public LoRA metadata: explicit fields only, never a project/config dump."""
import copy
import json
import math

SCHEMA_VERSION = 1
STUDIO_VERSION = "0.1.0"


def normalize_metadata(value=None):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("LoRA metadata must be an object")
    result = {}
    for key, limit in (("description", 4000), ("author", 200), ("license", 500)):
        text = value.get(key, "")
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError(f"Metadata {key} must be text up to {limit} characters")
        result[key] = text.strip()
    words = value.get("trigger_words")
    if words is not None:
        if not isinstance(words, list) or len(words) > 128:
            raise ValueError("Metadata trigger words must be a list of up to 128 phrases")
        cleaned = []
        for word in words:
            if not isinstance(word, str) or len(word) > 200:
                raise ValueError("Each metadata trigger phrase must be text up to 200 characters")
            word = word.strip()
            if word and word not in cleaned:
                cleaned.append(word)
        words = cleaned or None
    result["trigger_words"] = words
    strength = value.get("recommended_strength")
    if strength is not None and (isinstance(strength, bool) or not isinstance(strength, (int, float))
                                 or not math.isfinite(strength) or not -10 <= strength <= 10):
        raise ValueError("Recommended strength must be a finite number between -10 and 10, or blank")
    result["recommended_strength"] = strength
    return result


def public_metadata(project, process, model):
    """Select effective adapter settings; new config keys remain private by default."""
    def select(section, keys):
        return {key: copy.deepcopy(process[section][key]) for key in keys if key in process[section]}

    fields = normalize_metadata(project.get("lora_metadata"))
    fields["trigger_words"] = fields["trigger_words"] or ([project["trigger"]] if project["trigger"] else [])
    train = select("train", ("batch_size", "steps", "gradient_accumulation", "train_unet",
        "train_text_encoder", "gradient_checkpointing", "cache_text_embeddings", "noise_scheduler",
        "optimizer", "lr", "dtype", "disable_sampling", "timestep_type"))
    optimizer = process["train"].get("optimizer_params", {})
    if "weight_decay" in optimizer:
        train["optimizer_params"] = {"weight_decay": optimizer["weight_decay"]}
    dataset = process["datasets"][0]
    training = {
        "model": {"id": model["id"], "arch": model["arch"], "base_model": model["base"],
                  **select("model", ("quantize", "quantize_te", "qtype", "qtype_te"))},
        "network": select("network", ("type", "linear", "linear_alpha")),
        "train": train,
        "seed": project["training"]["seed"],
        "dataset": {key: copy.deepcopy(dataset[key]) for key in ("resolution", "caption_dropout_rate",
                    "shuffle_tokens", "cache_latents_to_disk", "buckets", "square_crop", "random_crop",
                    "random_scale", "flip_x", "flip_y") if key in dataset},
        "save": select("save", ("dtype", "save_every", "max_step_saves_to_keep")),
        "sample": select("sample", ("sampler", "sample_every", "width", "height", "seed",
                                   "walk_seed", "guidance_scale", "sample_steps")),
    }
    if model["id"] == "krea2":
        training["model"]["weight_source"] = project["training"].get("krea_weights", "original")
    public = {"schema_version": SCHEMA_VERSION, "studio_version": STUDIO_VERSION,
              "name": project["name"], **fields, "training": training}
    # AI Toolkit expands literal [name] in metadata. Escape its opening bracket
    # inside our JSON string so authored text survives that backend substitution.
    encoded = json.dumps(public, ensure_ascii=False, allow_nan=False).replace("[name]", "\\u005bname]")
    return {"lora_studio": encoded}
