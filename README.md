# Machine-Learning

งานนี้เป็นส่วนหนึ่งของวิชา **Machine-Learning**  
หัวข้อ: การหาค่า **Bias** และ **Variance** ด้วย **Analytical Method** และ **Simulation**

## โจทย์

ให้สุ่มข้อมูล 2 ตัวอย่างจากการแจกแจงแบบเอกรูป

\[
x_1, x_2 \sim Uniform[-1,1]
\]

แล้วกำหนด \(y_i=f(x_i)\) โดยทดลองกับฟังก์ชันเป้าหมาย 2 แบบ

1. \(f(x)=\sin(\pi x)\)
2. \(f(x)=x^2\)

เปรียบเทียบแบบจำลอง 3 แบบ

1. **Constant Model**: \(h(x)=b\)
2. **Linear Model**: \(h(x)=ax+b\)
3. **Linear Through Origin Model**: \(h(x)=ax\)

และสร้าง **Learning Curve** เปรียบเทียบทั้ง 3 แบบจำลอง พร้อมทดลองกรณีมี noise

---

## นิยาม Bias และ Variance

ให้ \(g_D(x)\) คือสมมติฐานที่ได้จากชุดข้อมูลสุ่ม \(D\)

\[
\bar{g}(x)=E_D[g_D(x)]
\]

\[
Bias=E_x[(\bar{g}(x)-f(x))^2]
\]

\[
Variance=E_x[E_D[(g_D(x)-\bar{g}(x))^2]]
\]

ดังนั้นค่า Expected Out-of-Sample Error คือ

\[
E_{out}=Bias+Variance
\]

---

## โครงสร้างไฟล์

```text
Machine-Learning/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── 00_models.py
│   ├── 01_analytical_method.py
│   ├── 02_simulation.py
│   ├── 03_learning_curve.py
│   └── 04_run_all.py
│
├── results/
│   ├── analytical_results.csv
│   ├── simulation_results.csv
│   └── learning_curve_results.csv
│
└── figures/
    ├── learning_curve_sin_pi_x_noise_0.00.png
    ├── learning_curve_sin_pi_x_noise_0.30.png
    ├── learning_curve_x_squared_noise_0.00.png
    └── learning_curve_x_squared_noise_0.30.png
```

---

## วิธีรันใน VS Code

สร้าง virtual environment

```bash
python -m venv .venv
```

เปิดใช้งาน virtual environment บน Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

ติดตั้ง library

```bash
pip install -r requirements.txt
```

รันงานทั้งหมด

```bash
python src/04_run_all.py
```

---

## หลักการสร้างแบบจำลอง

โปรเจกต์นี้ใช้ **normal equation / least squares** ผ่าน `numpy.linalg.lstsq`  
ไม่ได้ใช้ gradient descent

แบบจำลองแต่ละแบบใช้ design matrix ดังนี้

### 1. Constant Model

\[
h(x)=b
\]

\[
X=\begin{bmatrix}1\\1\\\vdots\\1\end{bmatrix}
\]

### 2. Linear Model

\[
h(x)=ax+b
\]

\[
X=\begin{bmatrix}x_1 & 1\\x_2 & 1\\\vdots & \vdots\\x_n & 1\end{bmatrix}
\]

### 3. Linear Through Origin Model

\[
h(x)=ax
\]

\[
X=\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}
\]

---

## ผลลัพธ์จาก Analytical Method

ค่าด้านล่างได้จากไฟล์ `src/01_analytical_method.py`

| Target Function | Model | \(\bar{g}(x)\) | Bias | Variance | Bias + Variance |
|---|---|---:|---:|---:|---:|
| \(\sin(\pi x)\) | Constant | 0 | 0.500000 | 0.250000 | 0.750000 |
| \(\sin(\pi x)\) | Linear | 0.775929x | 0.206717 | 1.676282 | 1.882999 |
| \(\sin(\pi x)\) | Through Origin | 1.428027x | 0.270644 | 0.236576 | 0.507219 |
| \(x^2\) | Constant | 0.333333 | 0.088889 | 0.044444 | 0.133333 |
| \(x^2\) | Linear | 0 | 0.200000 | 0.333333 | 0.533333 |
| \(x^2\) | Through Origin | 0x | 0.200000 | 0.114921 | 0.314921 |

---

## การตีความผล

### กรณี \(f(x)=\sin(\pi x)\)

แบบจำลอง **Linear Through Origin** ให้ค่า Error รวมต่ำที่สุดในกรณีสุ่มข้อมูล 2 จุด เพราะฟังก์ชัน \(\sin(\pi x)\) เป็นฟังก์ชันคี่ และเส้นตรงที่ผ่านจุดกำเนิดสามารถจับแนวโน้มหลักของฟังก์ชันได้ดี

แบบจำลอง **Linear Model** มี Bias ต่ำกว่า แต่ Variance สูงมาก เพราะเมื่อมีข้อมูลเพียง 2 จุด เส้นตรง \(ax+b\) จะลากผ่านสองจุดพอดี ทำให้โมเดลไวต่อข้อมูลสุ่มมาก

### กรณี \(f(x)=x^2\)

แบบจำลอง **Constant Model** ให้ค่า Error รวมต่ำที่สุด เพราะ \(x^2\) เป็นฟังก์ชันคู่ และเมื่อมองเฉลี่ยบนช่วง \([-1,1]\) ค่าคงที่ใกล้ค่าเฉลี่ยของ \(x^2\) สามารถประมาณภาพรวมได้ดี

แบบจำลอง **Linear Model** และ **Linear Through Origin** ไม่เหมาะกับ \(x^2\) เท่าค่าคงที่ในกรณีข้อมูลเพียง 2 จุด เพราะเส้นตรงไม่สามารถแทนความโค้งของพาราโบลาได้ดี

---

## ไฟล์ผลลัพธ์

หลังจากรัน `python src/04_run_all.py` จะได้ไฟล์ดังนี้

- `results/analytical_results.csv`  
  ผล Bias/Variance จาก analytical method

- `results/simulation_results.csv`  
  ผล Bias/Variance จาก Monte Carlo simulation

- `results/learning_curve_results.csv`  
  ผล train MSE และ test MSE สำหรับ learning curve

- `figures/*.png`  
  กราฟ learning curve เปรียบเทียบแบบจำลอง

---

## การอัปโหลดขึ้น GitHub

ถ้าสร้าง repository ใน GitHub แล้ว ให้ใช้คำสั่งนี้ในโฟลเดอร์โปรเจกต์

```bash
git init
git add .
git commit -m "add bias variance analytical simulation and learning curves"
git branch -M main
git remote add origin https://github.com/USERNAME/Machine-Learning.git
git push -u origin main
```

ให้เปลี่ยน `USERNAME` เป็น username GitHub ของตัวเอง

## Bias-Variance Decomposition Figure

โปรเจกต์นี้เพิ่มไฟล์ `src/05_bias_variance_visualization.py` สำหรับสร้างกราฟแบบแสดง Bias-Variance Decomposition คล้ายภาพประกอบในห้องเรียน

กราฟนี้อ่านแบบนี้:

- เส้นสีเขียว คือ target function จริง `f(x)`
- เส้นสีเทา คือ hypothesis หลายเส้นที่ได้จากการสุ่ม training set หลายครั้ง
- เส้นประสีแดง คือ average hypothesis หรือ `g_bar(x)`
- พื้นที่สีแดงโปร่งใส คือบริเวณการกระจายของ hypothesis รอบ `g_bar(x)` หรือภาพเชิงกราฟของ variance
- ระยะห่างระหว่างเส้นประสีแดงกับเส้นสีเขียว สื่อถึง bias ของโมเดล

รันเฉพาะกราฟนี้:

```bash
python src/05_bias_variance_visualization.py
```

ไฟล์ที่ได้:

```text
figures/bias_variance_decomposition_noise_0.20.png
results/bias_variance_decomposition_noise_0.20.csv
```

