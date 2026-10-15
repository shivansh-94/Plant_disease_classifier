
import os
import json
import torch
import timm
import streamlit as st
import pandas as pd

from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODEL_CONFIG = {
    "ResNet18": {
        "architecture": "resnet18",
        "file": "ResNet18.pth"
    },
    "EfficientNetV2-S": {
        "architecture": "tf_efficientnetv2_s",
        "file": "EfficientNetV2-S.pth"
    },
    "ConvNeXt-Tiny": {
        "architecture": "convnext_tiny",
        "file": "ConvNeXt-Tiny.pth"
    },
    "ViT-Small": {
        "architecture": "vit_small_patch16_224",
        "file": "ViT-Small.pth"
    }
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    CLASS_NAMES = json.load(f)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, config in MODEL_CONFIG.items():

        model_path = os.path.join(
            BASE_DIR,
            config["file"]
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model file: {config['file']}"
            )

        model = timm.create_model(
            config["architecture"],
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )

        try:
            state_dict = torch.load(
                model_path,
                map_location=DEVICE,
                weights_only=True
            )
        except TypeError:
            state_dict = torch.load(
                model_path,
                map_location=DEVICE
            )

        # Strip profiling variables if they exist in the loaded dictionary
        clean_state_dict = {
            k: v for k, v in state_dict.items()
            if "total_ops" not in k and "total_params" not in k
        }

        # Load with strict=False to gracefully bypass any remaining mismatched keys
        model.load_state_dict(clean_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()

        loaded_models[name] = model

    return loaded_models


# --------------------------------------------------
# STREAMLIT INTERFACE
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="ፑ",
    layout="wide"
)

st.title("ፑ Plant Disease Classification")
st.write(
    "Compare four deep learning models on the same plant leaf image."
)

st.info(
    "Upload a clear leaf image. Predictions depend on the dataset "
    "classes used during training."
)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 2])

    with left:
         st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
         )

    with right:

        if st.button("ፀ Predict with all 4 models"):

            try:
                models = load_models()

                input_tensor = transform(image).unsqueeze(0)

                results = []

                progress = st.progress(0)

                for index, (name, model) in enumerate(
                    models.items()
                ):

                    # Use the device on which the model was loaded
                    input_device = next(model.parameters()).device
                    batch = input_tensor.to(input_device)

                    with torch.inference_mode():
                        outputs = model(batch)
                        probabilities = torch.softmax(
                            outputs, dim=1
                        )[0]

                    confidence, predicted_index = torch.max(
                        probabilities, dim=0
                    )

                    results.append({
                        "Model": name,
                        "Predicted Class": CLASS_NAMES[
                            predicted_index.item()
                        ],
                        "Confidence (%)": round(
                            confidence.item() * 100, 2
                        )
                    })

                    progress.progress(
                        (index + 1) / len(models)
                    )

                results_df = pd.DataFrame(results)

                st.subheader("Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Model Confidence")

                chart_df = results_df.set_index("Model")[
                    ["Confidence (%)"]
                ]

                st.bar_chart(chart_df)

                counts = results_df["Predicted Class"].value_counts()

                st.subheader("Model Agreement")

                if counts.iloc[0] >= 3:
                    st.success(
                        f"Majority prediction: {counts.index[0]} "
                        f"({counts.iloc[0]} of 4 models agree)"
                    )
                elif counts.iloc[0] == 2:
                    st.warning(
                        f"Top prediction: {counts.index[0]} "
                        f"(2 of 4 models agree)"
                    )
                else:
                    st.warning(
                        "The models disagree. Check the uploaded "
                        "image and compare individual predictions."
                    )

                st.caption(
                    "Confidence is the model's softmax score, not a "
                    "guarantee of real-world correctness. This demo "
                    "is not a substitute for expert plant diagnosis."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
