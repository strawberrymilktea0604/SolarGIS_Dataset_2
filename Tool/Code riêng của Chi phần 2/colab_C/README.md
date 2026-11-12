# colab_C — tinh chỉnh trên ô cắt ortho + `yolov8s-obb` vòng 1–4

Tải cả thư mục này lên `MyDrive/colab_C/`, mở `chay_ortho_tinhchinh.ipynb` bằng Colab, Runtime → GPU,
chạy ô 1→6. `MyDrive/colab_B/` của đợt trước phải còn nguyên — ô 5 đọc `khung_nhan.zip` và `chia_khoi.csv`
ở đó, **không phải tải lại 634 MB**.

| File | Là gì | Cỡ |
|---|---|--:|
| `o_ortho.zip` | 247 ô cắt ortho (ảnh INFERNO đã xoay 270°) + nhãn OBB 4 lớp | 41,6 MB |
| `chia_o_ortho.csv` | vai của từng ô ở 5 vòng, cùng bộ chia khối với B3 | 24 KB |
| `goc_pt.zip` | 5 trọng số `yolo26s-obb_v{0..4}_s0` của B3, làm điểm khởi | 107,5 MB |
| `chay_ortho_tinhchinh.ipynb` | notebook | 17 KB |

Ước lượng T4: ô 4 (tinh chỉnh ortho) ≈ **1 giờ**; ô 5 (`yolov8s-obb` ×4 vòng) ≈ **4 giờ**.

---

## ① Vì sao tinh chỉnh trên ortho

Bài báo giữ cố định model, trọng số, ngưỡng rồi so hai hướng xử lý. Phản biện sẽ hỏi câu tiếp theo:
*model train trên khung, đương nhiên kém trên ortho — train lại trên ortho thì sao?* Chưa có câu trả lời.

Làm theo kiểu **tinh chỉnh chứ không train từ đầu**. Ortho chỉ có một tấm, 727 hộp ground truth, so với
12.390 hộp của nhánh khung; train mới rồi thua thì phản biện quy cho thiếu mẫu và phép so hỏng. Tinh chỉnh
thì luận điểm thành *đã cho nhánh ghép ưu thế sân nhà mà vẫn thua* — mạnh hơn hẳn. Nếu khoảng cách co lại
thì cũng biết sớm, và bài đổi khung thành "ghép ảnh đòi thêm một lượt train lại" chứ không chết.

## Bộ ô cắt được dựng thế nào

`ma/cat_o_ortho_train.py` (chạy ở máy, 26 s). Không chấm tay lại trên ortho:

1. Lấy **nhãn tay trên khung**. 497/598 khung align của 31M-2 vừa có nhãn tay vừa có tư thế camera.
2. Chiếu 4 góc mỗi hộp xuống mặt đất bằng đúng chuỗi của `quet_loi_anh_le.py`
   (pixel nhiệt → H → pixel RGB → tia ECEF → cắt DSM → lon/lat), **có bù trễ chụp từng ảnh** vì ortho được
   dựng có bù. → 2.723 hộp thô.
3. Gộp trùng ở 0,60 m — cùng ngưỡng hai nhánh của bài — giữ bản chiếu từ khung mà hộp gần tâm ảnh nhất
   (ít méo nghiêng nhất). → **727 hộp** : Single 490, Multi 172, Bypass 65, **MOC 0**.
4. Cắt cửa sổ 640×512 px ortho chồng 50 %, bỏ ô có dưới 30 % pixel khác 0, xám → chuẩn hoá MINMAX →
   INFERNO → xoay 270°. Đúng lượt D của `quet_ortho_oof.py`, nên **model thấy lúc train đúng thứ nó thấy
   lúc chấm**. → 247 ô, 2.867 dòng nhãn (một hộp nằm trong ~4 ô vì chồng 50 %).

**Vai từng vòng.** Ô tra khung TN19 có nhãn gần nhất trong `chia_khoi_dem_khung.csv` — cùng quy tắc láng
giềng với `quet_ortho_oof.py`. **Tâm ô và cả 4 góc ô phải cùng một vai**, lệch nhau thì ô thành `dem` và bị
bỏ. Không có ràng buộc đó thì ô vắt qua ranh giới khối sẽ kéo vùng test của vòng ấy vào train.
`val` dùng chính các ô `dem`: chúng không nằm trong test nên không rò rỉ vào con số của bài.

| vòng | ô train | ô val (`dem`) | ô test |
|---|--:|--:|--:|
| 0 | 139 | 64 | 44 |
| 1 | 133 | 114 | **0** |
| 2 | 115 | 108 | 24 |
| 3 | 116 | 116 | 15 |
| 4 | 128 | 67 | 52 |

**Cấu hình tinh chỉnh** giống hệt B3, đổi đúng hai chỗ: `epochs` 150 → 60 và `lr0` 0,005 → 0,001.
Mọi thứ khác (batch 16, imgsz 640, degrees 15, flipud/fliplr 0,5, mosaic 1,0, close_mosaic 15) giữ nguyên
để chỉ có một biến đổi.

## Năm chỗ hở phải ghi khi trích

1. **Nhãn ortho là nhãn chiếu, không phải nhãn chấm tay trên ortho.** Chúng thừa hưởng sai số của chuỗi
   chiếu và của phép ghép. Model tinh chỉnh vì thế học cả phần lệch ấy.
2. **Không có lớp ModuleOpenCircuit** trên ortho (cụm MOC nằm ở 42M, ngoài vùng ghép). Thực chất 3 lớp,
   `nc` vẫn để 4 cho khớp trọng số gốc.
3. **7,7 % dòng nhãn bị cắt ở mép ô** (222/2.867). Giữ và cắt chứ không bỏ: bỏ thì ô đó dạy model rằng
   chỗ có lỗi là chỗ không có lỗi, hại hơn.
4. **Vòng 1 không có ô test** — vùng bay của ortho không giao khối test vòng 1. Đúng như lượt quét
   out-of-fold hiện có: vòng đó không được dùng trên ortho.
5. **mAP trên ô test chỉ là tín hiệu nhanh.** Con số của bài là **cấp tấm** trên nền 562 tấm soi tay,
   phải chạy ở máy sau khi tải trọng số về.

## Sau khi Colab chạy xong

Tải về:

```
colab_C/ket_qua/runs_ortho/yolo26s-obb_v*_s0/weights/best.pt -> ket_qua/colab_C_ket_qua/runs_ortho/
colab_C/ket_qua/ortho_tc_*.json                              -> ket_qua/colab_C_ket_qua/
colab_B/ket_qua/yolov8s-obb_v*_s0.json + runs/               -> ket_qua/colab_B_ket_qua/
```

Rồi chạy ở máy — `quet_ortho_oof.py` **không phải sửa gì**, chỉ trỏ `--runs` sang thư mục mới:

```
py -3.13 ma/quet_ortho_oof.py --runs ket_qua/colab_C_ket_qua/runs_ortho \
       --out ket_qua/ortho_xoay270_tinhchinh_ds.json
py -3.13 ma/gan_tam_pin_4lop.py --nguon ket_qua/ortho_xoay270_tinhchinh_ds.json
```

So với hai mốc đã có ở `BAI_TOAN_1_PHAT_HIEN.md` §5f (cấp tấm, out-of-fold, nền 562):

| | P | R | F1 |
|---|:-:|:-:|:-:|
| nhánh khung | 0,886 | 0,924 | 0,905 |
| nhánh ortho, model khung | 0,936 | 0,834 | 0,882 |
| nhánh ortho, model tinh chỉnh | ? | ? | ? |

Câu hỏi là dòng ba có kéo R lên tới 0,924 không. ΔR hiện là +0,08…+0,10 và khoảng tin cậy không chứa 0.

## ② `yolov8s-obb` vòng 1–4

Bảng A đợt trước chỉ chạy **vòng 0**. Bốn lượt này cho `yolov8s-obb` đủ 5 vòng, tức có out-of-fold cho một
bộ dò thứ hai — để trả lời "ΔR là tính chất của ảnh hay của riêng `yolo26s-obb`". `patience` để 50 ngay từ
đầu (đợt trước 20 cắt mất hai hạt ở 56/43 epoch). Kết quả ghi thẳng về `MyDrive/colab_B/ket_qua` nên
`ma/gom_bangA_B3.py` gom được như cũ, không phải sửa.

Muốn có ΔR cho bộ dò thứ hai thì sau đó chạy thêm ở máy:
`ma/lap_model_theo_seq_b3.py` (đổi sang trọng số `yolov8s-obb`) → `quet_loi_anh_le.py --model-theo-seq`
cho nhánh khung, và `quet_ortho_oof.py --runs …` cho nhánh ortho.
