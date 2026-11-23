from ucimlrepo import fetch_ucirepo
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# fetch dataset
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, name), dpi=150)
    plt.close()

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

# Створення єдиного DataFrame (якщо потрібно)
# Можливо, вам доведеться налаштувати цей крок залежно від структури вашого датасету
if y is not None:
    df = pd.concat([X, y], axis=1)
else:
    df = X

# --- Етап 2: Ознайомлення з даними ---

# Виведення метаданих
print("\nМетадані:")
print(wine_quality.metadata)

# Виведення інформації про змінні
print("\nІнформація про змінні:")
print(wine_quality.variables)

# Виведення перших 5 рядків датасету
print("\nПерші 5 рядків:")
print(df.head())

# Виведення основних статистичних характеристик
print("\nОписова статистика:")
print(df.describe())


# --- Етап 3: Побудова графіків ---

# Важливо: Замініть 'column_name', 'x_column', 'y_column' тощо на реальні назви стовпців з вашого датасету.

# 1. Гістограма
# Дозволяє побачити розподіл значень однієї числової змінної.
plt.figure(figsize=(8, 5))
sns.histplot(df['alcohol'], bins=30, kde=True, color="darkred", label="alcohol")
plt.title("Розподіл вмісту алкоголю у винах")
plt.xlabel("Алкоголь, % об.")
plt.ylabel("Кількість зразків")
plt.legend()
save("histogram_alcohol.png")
# plt.show() # Розкоментуйте, якщо хочете побачити графік одразу

# 2. Діаграма стовпчикова (Bar plot)
# Показує кількість вин для кожної оцінки якості.
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="quality",hue="quality", palette="viridis",  legend=False)
plt.title("Кількість вин за оцінкою якості")
plt.xlabel("Якість (оцінка)")
plt.ylabel("Кількість зразків")
plt.legend(title="Колір вина")
save("bar_quality.png")
# plt.show()

# 3. Стовпчикова діаграма (Bar plot)
# Порівнює середнє значення числової змінної для різних категорій.
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Кореляційна матриця фізико-хімічних показників")
save("heatmap_corr.png")
# plt.show()

# 4. Ящик з вусами (Box plot)
# Показує розподіл даних, медіану, квартилі та викиди.
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="quality", y="alcohol", hue="quality",
            palette="viridis", legend=False)
plt.title("Вміст алкоголю залежно від якості")
plt.xlabel("Якість (оцінка)")
plt.ylabel("Алкоголь, % об.")
save("boxplot_alcohol_quality.png")
# plt.show()

# 5. Діаграма розсіювання: густина проти алкоголю
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="alcohol", y="density", alpha=0.4, color="goldenrod", label="вина")
plt.title("Залежність густини від вмісту алкоголю")
plt.xlabel("Алкоголь, % об.")
plt.ylabel("Густина, г/см³")
plt.legend()
save("scatter_density_alcohol.png")


# 6. Кругова діаграма: частка вин за групами якості
groups = pd.cut(df["quality"], bins=[0, 5, 6, 10],
                labels=["Низька (≤5)", "Середня (6)", "Висока (≥7)"])
counts = groups.value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%",  colors=["firebrick", "gold", "seagreen"], startangle=90)
plt.title("Частка вин за групами якості")
plt.legend(counts.index, title="Група якості")
save("pie_quality.png")


# --- Етап 4: Заповнення звіту ---
# Не забудьте додати збережені графіки та ваші висновки у файл REPORT.md!
