# CudaRayTracer

**CudaRayTracer** to projekt stworzony podczas studiów we współpracy dwóch autorów, mający na celu implementację ray tracera działającego na GPU z wykorzystaniem **CUDA**. Wspiera zaawansowane techniki renderingu oraz adaptacyjny antyaliasing, a całość została osadzona w nowoczesnym środowisku z użyciem C++20.

## ✨ Funkcje

- **Adaptacyjny Antyaliasing**
  - Piksele dzielone są na 4 części (boxy).
  - Każdy box może zostać podzielony rekurencyjnie (maks. 4 poziomy zagłębienia – ograniczenie CUDA).
  - Jeśli wszystkie 4 rogi boxa mają ten sam kolor – dalszy podział jest przerywany (early termination).

- **Global Illumination**
  - Realistyczne oświetlenie z odbiciami światła od innych obiektów.

- **Area Light z wykorzystaniem LTC (Linearly Transformed Cosines)**
  - Dokładne i efektywne światło powierzchniowe.

- **Materiały z obsługą koloru**
  - **Diffuse** – rozpraszający światło.
  - **Reflective** – odbijający światło (lustro).
  - **Refractive** – załamujący światło (szkło).
  - Każdy materiał może mieć własny kolor.

- **Scena**
  - Domyślna scena to **Cornell Box** z dwiema sferami:
    - Jedna **refrakcyjna**.
    - Druga **refleksyjna**.
  - Scena definiowana **bezpośrednio w kodzie źródłowym**.

- **Zapis do pliku**
  - Renderowany obraz zapisywany jest jako `file.hdr`.

## 🧰 Wykorzystane biblioteki

- [**SFML**](https://www.sfml-dev.org/) – obsługa okna i wyświetlania.
- [**stb_image_write**](https://github.com/nothings/stb) – zapis obrazu do pliku.
- [**stb_image**](https://github.com/nothings/stb) – wczytywanie tekstur (LTC).

📦 **Instalacja bibliotek przez vcpkg**

Wszystkie zewnętrzne biblioteki są pobierane i zarządzane za pomocą [**vcpkg**](https://github.com/microsoft/vcpkg):

- `sfml`
- `stb` (zawiera `stb_image` oraz `stb_image_write`)

Upewnij się, że masz skonfigurowane środowisko Visual Studio zintegrowane z vcpkg, np.:

```bash
vcpkg install sfml stb
```

W Visual Studio możesz ustawić vcpkg jako domyślny menedżer pakietów CMake/vcpkg lub dodać ścieżkę toolchain:
```
CMake toolchain file: [ścieżka_do_vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## ⚙️ Wymagania

- **Windows 10/11**
- **Visual Studio 2022**
- **CUDA Toolkit (12.9 zalecany)**
- **NVIDIA GPU z obsługą CUDA**
- **C++20** (kompilator MSVC)

## 🛠️ Kompilacja i uruchomienie

1. Otwórz projekt `CudaRayTracer.sln` w **Visual Studio 2022**.
2. Ustaw konfigurację na `Release` lub `Debug`.
3. Upewnij się, że środowisko korzysta z CUDA Runtime.
4. Uruchom (`Ctrl+F5`).

Po renderowaniu obraz zostanie zapisany do pliku `file.hdr`.

## 🖼️ Podgląd wyników

Przykład wynikowego obrazu (Cornell Box):
```
📁 file.hdr
```

Możesz go otworzyć np. w HDR viewerach, konwertować do `.png` lub innych formatów.

## 📂 Struktura projektu

```
.
├── src/                # Pliki źródłowe (CUDA + C++)
├── include/            # Nagłówki
├── file.hdr            # Wygenerowany obraz
├── external/           # Biblioteki zewnętrzne (stb, SFML)
├── CudaRayTracer.sln   # Projekt Visual Studio 2022
└── README.md
```

## 📌 Uwagi

- Scena nie jest ładowana z plików – modyfikacje odbywają się poprzez edycję kodu.
- Antyaliasing adaptacyjny ograniczony do 4 poziomów (CUDA).
- Wydajność renderingu zależy od użytej karty graficznej.

## 👥 Autorzy

Projekt stworzony wspólnie przez dwóch studentów w ramach nauki i eksploracji technik ray tracingu z użyciem CUDA.

## 📜 Licencja

Projekt udostępniony na licencji **MIT**.
