import flet as ft
import io
import base64
import matplotlib.pyplot as plt
import torch
from correspondences import find_correspondences, draw_correspondences

# Yardımcı fonksiyon: PIL Image'ı base64 stringe çevirir.
def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Yardımcı fonksiyon: matplotlib figürünü base64 stringe çevirir.
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def main(page: ft.Page):
    page.title = "Karşılık Noktaları Gösterimi"
    
    # Resim dosya yolları için değişkenler
    image1_path = None
    image2_path = None

    # FilePicker’lar
    file_picker1 = ft.FilePicker(on_result=lambda e: file_picker1_result(e))
    file_picker2 = ft.FilePicker(on_result=lambda e: file_picker2_result(e))
    page.overlay.append(file_picker1)
    page.overlay.append(file_picker2)

    # Yüklenen resimleri göstermek için flet Image kontrolleri
    img1 = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
    img2 = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
    # Karşılık noktaları çizilen figürler için Image kontrolleri
    result_img1 = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
    result_img2 = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)

    # FilePicker sonuçları alındığında, ilgili resim yolunu güncelleyen fonksiyonlar
    def file_picker1_result(e: ft.FilePickerResultEvent):
        nonlocal image1_path
        if e.files:
            image1_path = e.files[0].path
            # Resim dosyasını okuyup base64 formatına çevirme
            with open(image1_path, "rb") as f:
                img1.src_base64 = base64.b64encode(f.read()).decode("utf-8")
            page.update()

    def file_picker2_result(e: ft.FilePickerResultEvent):
        nonlocal image2_path
        if e.files:
            image2_path = e.files[0].path
            with open(image2_path, "rb") as f:
                img2.src_base64 = base64.b64encode(f.read()).decode("utf-8")
            page.update()

    def pick_image1(e):
        file_picker1.pick_files(allow_multiple=False)

    def pick_image2(e):
        file_picker2.pick_files(allow_multiple=False)

    # İşlem butonuna tıklandığında, karşılık noktaları hesaplama ve görsellerin güncellenmesi
    def process_images(e):
        if image1_path and image2_path:
            # Karşılık noktası hesaplaması için parametreler
            num_pairs = 10       # Görüntüdeki nokta eşleşme sayısı
            load_size = 224      # Görüntüyü yükleyeceğiniz boyut
            layer = 9            # Kullanılacak layer
            facet = 'key'        # Descriptor facet
            binned = True        # Binned descriptor kullanımı
            thresh = 0.05        # fg/bg threshold
            model_type = 'dino_vits8'  # Kullanılan model tipi
            stride = 4           # Stride değeri

            with torch.no_grad():
                points1, points2, image1_pil, image2_pil = find_correspondences(
                    image1_path, image2_path, num_pairs, load_size, layer,
                    facet, binned, thresh, model_type, stride)
            # matplotlib ile karşılık noktalarının çizimi
            fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
            
            # Oluşan figürleri base64 stringe dönüştürüp flet görseline aktaralım
            result_img1.src_base64 = fig_to_base64(fig1)
            result_img2.src_base64 = fig_to_base64(fig2)
            page.update()
        else:
            page.snack_bar = ft.SnackBar(ft.Text("Lütfen her iki resmi de seçin!"))
            page.snack_bar.open = True
            page.update()

    # Sayfa düzeni
    page.add(
        ft.Row(
            controls=[
                ft.Column(
                    controls=[
                        ft.Text("Resim 1"),
                        img1,
                        ft.ElevatedButton("Resim 1 Seç", on_click=pick_image1),
                    ]
                ),
                ft.Column(
                    controls=[
                        ft.Text("Resim 2"),
                        img2,
                        ft.ElevatedButton("Resim 2 Seç", on_click=pick_image2),
                    ]
                ),
            ]
        ),
        ft.ElevatedButton("Karşılık Noktalarını Hesapla", on_click=process_images),
        ft.Row(
            controls=[
                ft.Column(
                    controls=[
                        ft.Text("Karşılık Noktaları Görseli 1"),
                        result_img1,
                    ]
                ),
                ft.Column(
                    controls=[
                        ft.Text("Karşılık Noktaları Görseli 2"),
                        result_img2,
                    ]
                ),
            ]
        ),
    )

ft.app(target=main)
