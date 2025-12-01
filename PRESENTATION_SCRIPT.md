# SCRIPT THUYET TRINH: 3 MO HINH DEEP LEARNING CHO HE THONG GOI Y

---

## PHAN 1: GIOI THIEU TONG QUAN

### Slide mo dau

"Xin chao thay/co va cac ban. Hom nay em se trinh bay ve 3 mo hinh Deep Learning ma em da xay dung cho he thong goi y san pham thoi trang.

Ba mo hinh nay deu dua tren kien truc BiLSTM (Bidirectional Long Short-Term Memory), nhung khac nhau o co che tong hop thong tin:
1. **Deep BiLSTM** - Su dung Mean Pooling don gian
2. **BiLSTM + Self-Attention** - Su dung co che Attention tu hoc trong so
3. **BiLSTM + Multi-Head Attention** - Su dung co che Attention da dau kieu Transformer

Muc tieu cua cac mo hinh la: Du doan bai post tiep theo ma nguoi dung se tuong tac, dua tren lich su tuong tac truoc do."

---

## PHAN 2: DU LIEU DAU VAO

### Slide: Input Data

"Truoc khi di vao chi tiet tung mo hinh, em xin trinh bay ve du lieu dau vao:

**Input**: Chuoi cac Post ID ma nguoi dung da tuong tac
- Vi du: User A da tuong tac voi cac post: [Post_15, Post_42, Post_8, Post_103]
- Chuoi nay duoc ma hoa thanh cac chi so: [15, 42, 8, 103]

**Output**: Du doan Post ID tiep theo ma nguoi dung se tuong tac
- Vi du: Mo hinh du doan Post_67 la post tiep theo

**Cac thong so:**
- Vocabulary Size: 530 posts (tong so bai post trong he thong)
- Embedding Dimension: 64
- Hidden Dimension: 64
- Batch Size: 32
- Sequence Length: Toi da 20 tuong tac gan nhat"

---

## PHAN 3: MO HINH 1 - DEEP BiLSTM

### Slide: Kien truc Deep BiLSTM

"Mo hinh dau tien la Deep BiLSTM - day la mo hinh baseline don gian nhat.

**Buoc 1: Embedding Layer**
```
Input: [Batch, Seq_len] - Vi du: [32, 10] (32 users, moi user 10 tuong tac)
Output: [Batch, Seq_len, 64] - Vi du: [32, 10, 64]
```
- Moi Post ID duoc chuyen thanh vector 64 chieu
- Su dung padding_idx=0 de xu ly cac chuoi co do dai khac nhau

**Buoc 2: Bidirectional LSTM**
```
Input: [32, 10, 64]
Output: [32, 10, 128] (64*2 do bidirectional)
```
- LSTM doc chuoi tu TRAI SANG PHAI va tu PHAI SANG TRAI
- Ket hop ca 2 huong giup mo hinh hieu ngur canh tot hon
- Output dimension = 64 * 2 = 128

**Buoc 3: Mean Pooling**
```
Input: [32, 10, 128]
Output: [32, 128]
```
- Lay TRUNG BINH cua tat ca cac timestep
- Cong thuc: output = mean(lstm_out, dim=1)
- Uu diem: Don gian, nhanh
- Nhuoc diem: Tat ca timestep duoc doi xu NGANG BANG, khong phan biet timestep nao quan trong hon

**Buoc 4: Dropout + Fully Connected**
```
Dropout(0.5): Ngan chan overfitting
FC Layer: [32, 128] -> [32, 530]
```
- Output la xac suat cho 530 posts
- Chon post co xac suat cao nhat de goi y"

### Slide: Nhuoc diem cua Mean Pooling

"Tai sao Mean Pooling chua tot?

Gia su nguoi dung co lich su: [Ao_so_mi, Quan_jean, Giay_the_thao, Tui_xach]

Voi Mean Pooling, tat ca 4 san pham duoc coi la QUAN TRONG NHU NHAU.

Nhung thuc te, tuong tac GAN NHAT (Tui_xach) co the quan trong hon trong viec du doan san pham tiep theo.

=> Day la ly do chung ta can co che ATTENTION."

---

## PHAN 4: MO HINH 2 - BiLSTM + SELF-ATTENTION

### Slide: Kien truc Self-Attention

"Mo hinh thu hai them co che Self-Attention de hoc TRONG SO cho moi timestep.

**Buoc 1-2: Giong Deep BiLSTM**
- Embedding -> BiLSTM -> [Batch, Seq, 128]

**Buoc 3: Tinh Attention Score**
```python
attn_scores = Linear(128, 1)(lstm_out)  # [Batch, Seq, 1]
```
- Moi timestep duoc gan 1 diem so (score)
- Score cao = timestep quan trong

**Buoc 4: Softmax Normalization**
```python
attn_weights = softmax(attn_scores, dim=1)  # [Batch, Seq, 1]
```
- Chuyen score thanh xac suat (tong = 1)
- Vi du: [0.1, 0.15, 0.25, 0.5] - Timestep cuoi quan trong nhat

**Buoc 5: Weighted Sum (Context Vector)**
```python
context = sum(attn_weights * lstm_out, dim=1)  # [Batch, 128]
```
- Nhan trong so voi output LSTM
- Tong hop thanh 1 vector dai dien

**Buoc 6: Dropout + FC -> Output**"

### Slide: Minh hoa Attention

"Minh hoa cu the:

Lich su nguoi dung: [Ao_so_mi, Quan_jean, Giay_the_thao, Tui_xach]

**Sau khi hoc, Attention weights co the la:**
- Ao_so_mi: 0.10 (10%)
- Quan_jean: 0.15 (15%)
- Giay_the_thao: 0.25 (25%)
- Tui_xach: 0.50 (50%)

=> Mo hinh TU HOC rang tuong tac gan day (Tui_xach) quan trong hon.

**Cong thuc Context Vector:**
```
context = 0.10*embed(Ao) + 0.15*embed(Quan) + 0.25*embed(Giay) + 0.50*embed(Tui)
```

Day la diem khac biet lon so voi Mean Pooling!"

---

## PHAN 5: MO HINH 3 - BiLSTM + MULTI-HEAD ATTENTION

### Slide: Kien truc Multi-Head Attention

"Mo hinh thu ba su dung Multi-Head Attention - co che attention manh me nhat, lay cam hung tu kien truc Transformer.

**Diem khac biet chinh:**

1. **Nhieu Attention Heads (4 heads)**
   - Thay vi 1 attention, chung ta co 4 attention hoat dong SONG SONG
   - Moi head hoc 1 khia canh khac nhau cua du lieu
   - Head 1: Co the hoc ve LOAI san pham
   - Head 2: Co the hoc ve THUONG HIEU
   - Head 3: Co the hoc ve GIA CA
   - Head 4: Co the hoc ve THOI GIAN tuong tac

2. **Self-Attention voi Q, K, V**
```python
Q = K = V = lstm_output
attn_out = MultiheadAttention(Q, K, V)
```
   - Query (Q): 'Toi dang tim gi?'
   - Key (K): 'Cac timestep chua thong tin gi?'
   - Value (V): 'Gia tri thuc su cua moi timestep'

3. **Residual Connection + Layer Normalization**
```python
out = LayerNorm(attn_out + lstm_out)
```
   - Cong truc tiep input voi output (skip connection)
   - Giup gradient chay tot hon khi train
   - Giu lai thong tin goc khong bi mat"

### Slide: Chi tiet Multi-Head Attention

"Chi tiet ve 4 Attention Heads:

**Cau hinh:**
- embed_dim = 128 (output cua BiLSTM)
- num_heads = 4
- head_dim = 128 / 4 = 32 moi head

**Quy trinh:**
1. Chia input 128-dim thanh 4 phan, moi phan 32-dim
2. Moi head tinh attention doc lap
3. Ghep 4 output lai thanh 128-dim
4. Di qua 1 lop Linear de tong hop

**Tai sao nhieu head tot hon 1 head?**
- Mo hinh co the hoc NHIEU MOI QUAN HE cung luc
- Vi du thuc te: Khi mua sam, ban xem xet NHIEU YEU TO:
  - Phong cach (casual, formal)
  - Mau sac (toi, sang)
  - Gia ca (re, dat)
  - Thuong hieu (Nike, Adidas)
- Moi head co the chuyen mon hoa cho 1 yeu to"

### Slide: Residual Connection

"Tai sao can Residual Connection?

**Van de Vanishing Gradient:**
- Khi mo hinh sau, gradient co the bi 'bien mat' khi lan truyen nguoc
- Cac lop dau khong hoc duoc gi

**Giai phap: Skip Connection**
```
output = LayerNorm(attention_output + lstm_output)
```

**Loi ich:**
1. Gradient co duong di tat, khong bi mat
2. Mo hinh co the hoc 'su thay doi' thay vi hoc tu dau
3. Training on dinh hon, hoi tu nhanh hon

Day la ky thuat quan trong tu ResNet va Transformer!"

---

## PHAN 6: SO SANH 3 MO HINH

### Slide: Bang so sanh

"Bay gio em xin so sanh 3 mo hinh:

| Tieu chi | Deep BiLSTM | Self-Attention | Multi-Head Attention |
|----------|-------------|----------------|---------------------|
| Co che tong hop | Mean Pooling | Weighted Sum | MHA + Residual |
| So Attention Heads | 0 | 1 | 4 |
| Hoc trong so | Khong | Co | Co (da chieu) |
| Residual Connection | Khong | Khong | Co |
| Layer Normalization | Khong | Khong | Co |
| Do phuc tap | Thap | Trung binh | Cao |
| So parameters | ~168,850 | ~168,979 | ~235,154 |"

### Slide: Ket qua thuc nghiem

"Ket qua tren tap test:

| Mo hinh | HR@5 | HR@10 | MRR |
|---------|------|-------|-----|
| Deep BiLSTM | 0.1688 | 0.2165 | 0.1411 |
| Self-Attention | 0.2165 | 0.2597 | 0.1718 |
| **Multi-Head Attention** | **0.2208** | **0.2597** | **0.1870** |

**Giai thich cac chi so:**
- **HR@K (Hit Rate)**: Ty le ma san pham dung nam trong top K goi y
- **MRR (Mean Reciprocal Rank)**: Trung binh nghich dao hang cua san pham dung

**Phan tich:**
1. Self-Attention tot hon Deep BiLSTM:
   - HR@10 tang 20% (0.2165 -> 0.2597)
   - MRR tang 21.7% (0.1411 -> 0.1718)
   => Co che attention giup mo hinh hoc duoc timestep quan trong

2. Multi-Head Attention tot nhat:
   - MRR cao nhat (0.1870) - San pham dung duoc xep hang cao hon
   - Du HR@10 bang Self-Attention, nhung CHAT LUONG ranking tot hon"

---

## PHAN 7: KET LUAN

### Slide: Tong ket

"Tong ket lai:

**1. Deep BiLSTM:**
- Don gian, de implement
- Phu hop lam baseline
- Nhuoc diem: Khong phan biet duoc timestep quan trong

**2. BiLSTM + Self-Attention:**
- Them kha nang hoc trong so cho moi timestep
- Cai thien dang ke so voi baseline
- Phu hop khi can balance giua hieu suat va do phuc tap

**3. BiLSTM + Multi-Head Attention:**
- Kien truc manh me nhat
- Hoc duoc nhieu khia canh cua du lieu
- Residual connection giup training on dinh
- **=> MO HINH DUOC CHON CHO HE THONG**

**Huong phat trien:**
- Tang so layers cua BiLSTM
- Thu nghiem voi nhieu attention heads hon (8, 16)
- Ket hop voi thong tin noi dung san pham (content-based)
- Ap dung Transformer encoder thay BiLSTM"

### Slide: Cam on

"Em xin chan thanh cam on thay/co va cac ban da lang nghe.

Em san sang tra loi cac cau hoi."

---

## PHU LUC: CAC CAU HOI THUONG GAP

### Q1: Tai sao chon BiLSTM thay vi LSTM thuong?

"BiLSTM (Bidirectional) doc chuoi tu 2 huong:
- Forward: [A, B, C, D] -> Hieu ngur canh tu qua khu
- Backward: [D, C, B, A] -> Hieu ngur canh tu tuong lai

Trong bai toan goi y, viec nhin ca 2 huong giup hieu toan dien hon ve so thich nguoi dung."

### Q2: Tai sao dropout = 0.5, co qua cao khong?

"Dropout 0.5 la gia tri pho bien, co nghia la 50% neuron bi tat ngau nhien khi training.
- Giup ngan chan overfitting
- Buoc mo hinh hoc cac dac trung robust hon
- Khi inference, tat ca neuron duoc su dung

Trong thuc te, 0.3-0.5 la khoang thong dung. Co the tune hyperparameter de tim gia tri tot nhat."

### Q3: Multi-Head Attention co 4 heads, tai sao khong nhieu hon?

"So heads phu thuoc vao:
- embed_dim: 128 / 4 = 32 moi head (hop ly)
- Neu 8 heads: 128 / 8 = 16 moi head (co the qua nho)
- Can balance giua so heads va dimension moi head

4 heads la lua chon can bang cho embed_dim = 128."

### Q4: Loss function la gi?

"Su dung CrossEntropyLoss - pho bien cho bai toan phan loai nhieu lop.
- Input: Logits [Batch, 530]
- Target: Post ID dung
- Loss = -log(probability cua class dung)

Optimizer: Adam voi learning rate = 0.001"

### Q5: Tai sao MRR quan trong?

"MRR (Mean Reciprocal Rank) do CHAT LUONG ranking:
- Neu san pham dung o vi tri 1: score = 1/1 = 1.0
- Neu san pham dung o vi tri 2: score = 1/2 = 0.5
- Neu san pham dung o vi tri 5: score = 1/5 = 0.2

MRR cao = San pham dung thuong nam o TOP DAU danh sach goi y
=> Nguoi dung khong can cuon xuong nhieu de tim san pham muon"

---

## CODE MINH HOA

### Forward pass cua Multi-Head Attention:

```python
def forward(self, x):
    # x: [Batch, Seq_len] - Chuoi Post IDs

    # Buoc 1: Embedding
    embedded = self.embedding(x)  # [Batch, Seq, 64]

    # Buoc 2: BiLSTM
    lstm_out, _ = self.lstm(embedded)  # [Batch, Seq, 128]

    # Buoc 3: Multi-Head Attention (Q=K=V=lstm_out)
    attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)  # [Batch, Seq, 128]

    # Buoc 4: Residual + LayerNorm
    out = self.norm(attn_out + lstm_out)  # [Batch, Seq, 128]

    # Buoc 5: Mean Pooling
    context = torch.mean(out, dim=1)  # [Batch, 128]

    # Buoc 6: Dropout + FC
    out = self.dropout(context)  # [Batch, 128]
    return self.fc(out)  # [Batch, 530]
```

---

*Script nay duoc viet cho muc dich thuyet trinh luan van/do an.*
