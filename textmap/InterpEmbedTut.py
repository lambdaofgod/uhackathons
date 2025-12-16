# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: textmap
#     language: python
#     name: textmap
# ---

# %%
import interp_embed

# %%
from interp_embed import Dataset
from interp_embed.sae.local_sae import GoodfireSAE, LocalSAE
import pandas as pd

# 1. Load a Goodfire SAE or SAE supported through the SAELens package
sae = LocalSAE(
    release="gemma-scope-2b-pt-res",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="layer_19/width_16k/average_l0_279",
    device="cuda:0", # optional
)

goodfire_sae = GoodfireSAE(
    variant_name="Llama-3.1-8B-Instruct-SAE-l19",  # or "Llama-3.3-70B-Instruct-SAE-l50" for higher quality features
    device="cuda:0", # optional
    quantize=True
)

# %%
# %%time

# 2. Prepare your data as a DataFrame
df = pd.DataFrame({
    "text": ["Good morning!", "Hello there!", "Good afternoon."],
    "date": ["2022-01-10", "2021-08-23", "2023-03-14"]
})

# 3. Create dataset - computes and saves feature activations
dataset = Dataset(
    data=df,
    sae=goodfire_sae,
    field="text",  # Optional. Column containing text to analyze
    save_path="my_dataset.pkl"  # Optional. Auto-saves progress, which enables recovery if computations fail
)

# %%
labels = dataset.feature_labels()

# %%
annotated_document = dataset[0].token_activations(feature = 790)
print(annotated_document)

# %%
import numpy as np
import pandas as pd

def diff_features(
    ds1,
    ds2,
    metric="absolute",
    min_coverage=0.0,
    max_coverage=1.0
):
    """
    Calculate the differences in feature frequency (i.e. how often a feature is activated) between two datasets.

    :param other: Another DatasetFeatureActivations instance
    :param metric: Metric to use for calculating the difference ('absolute', 'relative')
    :param min_coverage: Minimum percentage of samples that must have a non-zero activation to consider a feature
    :param max_coverage: Maximum percentage of samples that must have a non-zero activation to consider a feature
    :return: Pandas DataFrame with feature labels and their frequency differences
    """
    # Get per-feature activations (frequency of nonzero activation for each feature)
    fa1 = ds1.latents("binarize")  # shape: (D1, F)
    fa2 = ds2.latents("binarize")  # shape: (D2, F)

    freq1 = np.sum(fa1, axis=0) / fa1.shape[0]
    freq2 = np.sum(fa2, axis=0) / fa2.shape[0]

    # Mask features outside coverage bounds
    min_freq = np.minimum(freq1, freq2)
    max_freq = np.maximum(freq1, freq2)
    mask = (min_freq < min_coverage) | (max_freq > max_coverage)
    freq1_masked = freq1.copy()
    freq2_masked = freq2.copy()
    freq1_masked[mask] = -1
    freq2_masked[mask] = -1

    # Calculate the difference according to metric
    if metric == "absolute":
        diff = freq1_masked - freq2_masked
    elif metric == "relative":
        denom = np.maximum(freq1_masked, freq2_masked).copy()
        denom[denom == 0] = 1  # avoid division by zero
        diff = (freq1_masked / denom) - (freq2_masked / denom)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Get feature labels
    feature_labels_dict = ds1.feature_labels()
    num_features = freq1.shape[0]
    feature_labels = [feature_labels_dict.get(i, "") for i in range(num_features)]

    df = pd.DataFrame({
        "feature": feature_labels,
        "feature_id": np.arange(num_features),
        ds1.id: freq1_masked,
        ds2.id: freq2_masked,
        "frequency_difference": diff,
    })

    df = df.sort_values(by=["frequency_difference", f"{ds1.id}"], ascending=False, ignore_index=True)
    return df


# %%
# !ls

# %% [markdown]
# ## Comparing Vic3 and EU5 reviews

# %%
with open("data/eu5_review.txt") as f:
    sections = f.read().split("##")
    section_headers = [s.split("\n")[0] for s in sections]
    eu5_texts = pd.DataFrame({"text": sections, "section_header": section_headers})

# %%
with open("data/vic3_review.txt") as f:
    sections = f.read().split("##")
    section_headers = [s.split("\n")[0] for s in sections]
    vic3_texts = pd.DataFrame({"text": sections, "section_header": section_headers})

# %%
# %%time
eu5_dataset = Dataset(
    data=eu5_texts,
    sae=goodfire_sae,
    field="text",  # Optional. Column containing text to analyze
    save_path="eu5_review_dataset.pkl"  # Optional. Auto-saves progress, which enables recovery if computations fail
)

# %%
vic3_dataset = Dataset(
    data=vic3_texts,
    sae=goodfire_sae,
    field="text",  # Optional. Column containing text to analyze
    save_path="vic3_review_dataset.pkl"  # Optional. Auto-saves progress, which enables recovery if computations fail
)

# %%
eu5_vic3_diff = diff_features(eu5_dataset, vic3_dataset, min_coverage=0.25, max_coverage=0.75)

# %%
eu5_vic3_diff
