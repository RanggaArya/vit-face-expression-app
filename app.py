import streamlit as st
import os
import torch
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import types
import requests

# ==== FUNGSI UNTUK DOWNLOAD MODEL DARI GOOGLE DRIVE ====
def download_file_from_google_drive(id, destination):
    """Mengunduh file dari Google Drive, menangani konfirmasi file besar."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Coba request pertama
    response = session.get(URL, params={'id': id}, stream=True)
    
    # Dapatkan token konfirmasi jika ada halaman peringatan
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # Jika ada token, lakukan request kedua dengan konfirmasi
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Tampilkan progress bar saat download
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = st.progress(0, text="Mengunduh model, mohon tunggu...")
    
    with open(destination, "wb") as f:
        bytes_downloaded = 0
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size > 0:
                    progress = int((bytes_downloaded / total_size) * 100)
                    progress_bar.progress(progress, text=f"Mengunduh model... {progress}%")

    progress_bar.empty() # Hapus progress bar setelah selesai
    st.success("Model berhasil diunduh!")

# ==== Konfigurasi Awal ====
st.set_page_config(
    page_title="Klasifikasi Ekspresi Wajah Majemuk",
    page_icon="😊",
    layout="wide"
)

# ==== PERIKSA DAN UNDUH MODEL JIKA PERLU ====
MODEL_PATH = "vit_rafdb_compound_cbce_best.pth"
if not os.path.exists(MODEL_PATH):
    # ==============================================================================
    # == PASTIKAN ANDA MENGGANTI FILE_ID DI BAWAH INI DENGAN YANG SUDAH BENAR      ==
    # ==============================================================================
    FILE_ID = "1_3LTlRaAezxfQg1DO7JrNY9WoWJCLEVE" 
    
    # Cek jika placeholder belum diganti
    if FILE_ID == "1_3LTlRaAezxfQg1DO7JrNY9WoWJCLEVE":
        st.error("Kesalahan Konfigurasi: Harap masukkan FILE_ID Google Drive Anda di dalam kode app.py.")
        st.stop()

    with st.spinner(f"File model tidak ditemukan. Mengunduh dari Google Drive..."):
        download_file_from_google_drive(FILE_ID, MODEL_PATH)


# Sisa kode Anda dari sini ke bawah tetap sama persis
# Pastikan kode di bawah ini ada di file Anda
label_names = [
    'Happily Surprised', 'Happily Disgusted', 'Sadly Surprised', 'Sadly Disgusted',
    'Sadly Fearful', 'Sadly Angry', 'Fearfully Surprised', 'Fearfully Angry',
    'Fearfully Disgusted', 'Angrily Surprised', 'Angrily Disgusted'
]
device = torch.device("cpu")

@st.cache_resource
def load_model_and_patcher():
    attention_weights_storage = []
    def new_attention_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attention_weights_storage.clear()
        attention_weights_storage.append(attn.detach().cpu())
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=len(label_names)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    last_attn_block = model.blocks[-1].attn
    last_attn_block.forward = types.MethodType(new_attention_forward, last_attn_block)
    return model, attention_storage

model, attention_storage = load_model_and_patcher()

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def predict_and_visualize(input_img_pil, model, attention_weights_storage):
    attention_weights_storage.clear()
    input_img_np = np.array(input_img_pil)
    resized_img = A.Resize(224, 224)(image=input_img_np)['image']
    processed_tensor = transform(image=input_img_np)["image"]
    input_batch = processed_tensor.unsqueeze(0).to(device)
    fig_patches = Figure(figsize=(4, 4))
    ax = fig_patches.subplots()
    ax.imshow(resized_img)
    ax.set_title("2. Image Split into Patches")
    patch_size = model.patch_embed.patch_size[0]
    ax.set_xticks(np.arange(0, resized_img.shape[1], patch_size), minor=False)
    ax.set_yticks(np.arange(0, resized_img.shape[0], patch_size), minor=False)
    ax.grid(which="major", color="yellow", linestyle="-", linewidth=1)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    with torch.no_grad():
        logits = model(input_batch)
    if not attention_weights_storage:
        st.error("Gagal mengekstrak attention weights. Terjadi kesalahan pada proses internal.")
        st.stop()
    att_mat = torch.mean(attention_weights_storage[0][0], dim=0)
    cls_att = att_mat[0, 1:].reshape(14, 14)
    cls_att_resized = F.interpolate(
        cls_att.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze()
    fig_attn = Figure(figsize=(5, 4))
    ax_attn = fig_attn.subplots()
    ax_attn.imshow(resized_img)
    im = ax_attn.imshow(cls_att_resized, cmap='inferno', alpha=0.6)
    ax_attn.set_title("3. ViT Attention Map")
    ax_attn.axis('off')
    fig_attn.colorbar(im, ax=ax_attn)
    probabilities = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    fig_probs = Figure(figsize=(6, 5))
    ax_probs = fig_probs.subplots()
    sns.barplot(x=probabilities * 100, y=label_names, ax=ax_probs, orient='h', palette="viridis", hue=label_names, legend=False)
    ax_probs.set_title("4. Final Classification Result")
    ax_probs.set_xlabel("Probability (%)")
    ax_probs.set_xlim(0, 100)
    for i, p in enumerate(probabilities):
        ax_probs.text(p * 100 + 1, i, f"{p*100:.2f}%", va='center')
    fig_probs.tight_layout()
    pred_idx = probabilities.argmax()
    pred_label = label_names[pred_idx]
    pred_prob = probabilities[pred_idx]
    final_prediction_text = f"### **Prediksi:** {pred_label}\n### **Keyakinan:** {pred_prob:.2%}"
    return resized_img, fig_patches, fig_attn, final_prediction_text, fig_probs

st.title("Klasifikasi Ekspresi Wajah Majemuk dengan Vision Transformer (ViT)")
st.markdown(
    """
    Aplikasi ini mendemonstrasikan bagaimana model ViT yang telah dilatih pada dataset RAF-DB Compound melakukan klasifikasi langkah demi langkah.
    Unggah gambar wajah untuk melihat prosesnya.
    """
)
uploaded_file = st.file_uploader("🖼️ Unggah Gambar Wajah", type=['jpg', 'jpeg', 'png'])
if uploaded_file is None:
    st.info("Silakan unggah sebuah gambar untuk memulai analisis.")
else:
    input_img_pil = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(input_img_pil, caption="Gambar Asli yang Diunggah", use_container_width=True)
    with col2:
        with st.spinner("🚀 Model sedang menganalisis gambar..."):
            resized_img, fig_patches, fig_attn, final_prediction_text, fig_probs = predict_and_visualize(input_img_pil, model, attention_storage)
            st.markdown("---")
            st.markdown("✅ **Hasil Prediksi Akhir**")
            st.markdown(final_prediction_text)
            st.markdown("---")
    with st.expander("Lihat Detail Proses Visualisasi 👇", expanded=True):
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.image(resized_img, caption="1. Gambar Input (Setelah Resize 224x224)", use_container_width=True)
            st.pyplot(fig_patches)
        with viz_col2:
            st.pyplot(fig_attn)
            st.pyplot(fig_probs)
