import streamlit as st
from PIL import Image
from torchvision import transforms

from cnn import predict as predict_with_cnn
from mlp import predict as predict_with_mlp


st.set_page_config(
    page_title="MNIST Handwritten Digit Recognition",
    page_icon="✍️",
)

st.title("MNIST 手写数字识别")
st.write("上传一张只包含一个手写数字的 PNG 或 JPG 图片。")

model_name = st.selectbox(
    "选择模型",
    options=["CNN", "MLP"],
    help="CNN 对图片空间结构的识别通常比 MLP 更好。",
)

uploaded_file = st.file_uploader(
    "选择图片",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="上传的原图", width=350)

    if st.button("开始识别", type="primary"):
        predictor = predict_with_cnn if model_name == "CNN" else predict_with_mlp
        prediction, confidence, image_batch, was_inverted = predictor(uploaded_image)
        processed_image = transforms.ToPILImage()(image_batch.squeeze(0))

        prediction_column, confidence_column = st.columns(2)
        prediction_column.metric("预测数字", prediction)
        confidence_column.metric("置信度", f"{confidence:.2%}")

        st.image(
            processed_image,
            caption=f"{model_name} 实际接收的 28×28 图片",
            width=280,
        )
        st.caption(f"自动反色：{'是' if was_inverted else '否'}")
