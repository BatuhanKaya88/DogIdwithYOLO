import os

# Etiket dosyalarının bulunduğu klasörler
label_dirs = [
    r"C:\Users\Sibel Kaya\Desktop\DogIdwithYOLO\.venv\dataset\labels\train",
    r"C:\Users\Sibel Kaya\Desktop\DogIdwithYOLO\.venv\dataset\labels\valid",
    r"C:\Users\Sibel Kaya\Desktop\DogIdwithYOLO\.venv\dataset\labels\test",
]

# Dosya yolunu inceleyerek etiketi belirleme
for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"Uyarı: {label_dir} bulunamadı.")
        continue

    # Dosya isimlerini kontrol et
    for folder in os.listdir(label_dir):
        folder_path = os.path.join(label_dir, folder)

        # Eğer klasör 'dog' ya da 'human' içeriyorsa etiketini belirle
        if os.path.isdir(folder_path):
            label = None
            if 'dog' in folder.lower():
                label = 0  # Dog etiketini 0 yap
            elif 'human' in folder.lower():
                label = 1  # Human etiketini 1 yap

            if label is not None:
                # Bu klasördeki .txt dosyalarını kontrol et
                for filename in os.listdir(folder_path):
                    if filename.endswith(".txt"):
                        filepath = os.path.join(folder_path, filename)

                        # Dosya yoksa silme işlemi yapılır
                        if not os.path.exists(filepath):
                            print(f"Uyarı: {filepath} bulunamadı.")
                            continue  # Bu dosya geçilir

                        with open(filepath, "r") as f:
                            lines = f.readlines()

                        # Her satırdaki sınıf ID'sini güncelle
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                parts[0] = str(label)  # Her satırdaki sınıf ID'sini güncelle
                                new_lines.append(" ".join(parts))

                        # Yeni düzenlenmiş satırları dosyaya yaz
                        with open(filepath, "w") as f:
                            f.write("\n".join(new_lines) + "\n")

print("Etiketler başarıyla dosya isimlerine göre güncellendi!")
