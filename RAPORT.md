# Raport projektu RIONA (C++)

## 1. Opis zadania projektowego
Celem projektu bylo zaimplementowanie algorytmow klasyfikacji:
- **RIONA** (reguly + lokalne sasiedztwo k-NN),
- **RIA** (klasyfikacja oparta wylacznie o reguly),
- **k+NN** (wariant k-NN z lokalna metryka SVDM),

z wykorzystaniem miary odleglosci dla atrybutow nominalnych **SVDM** lub **SVDM'**. Program wykonuje klasyfikacje w trybie **leave-one-out**, obsluguje tryb **globalny** i **lokalny** wyznaczania statystyk, a wyniki zapisuje do plikow OUT/STAT/kNN.

## 2. Przyjete zalozenia
- Dane wejsciowe sa w formacie **ARFF** (sekcje @RELATION, @ATTRIBUTE, @DATA).
- **Ostatni atrybut** w pliku jest atrybutem decyzyjnym (klasa).
- Typy atrybutow sa wyznaczane z ARFF:
  - numeric/real/integer -> **Numeric**,
  - string/nominal/date lub {lista} -> **Nominal**.
- Typy mozna recznie nadpisac parametrem `--types` (np. `n,c,n`).
- Braki danych oznaczone sa `?` (lub tokenem z `--missing`).
- Odleglosci dla brakow:
  - nominalne: **2** dla SVDM, **1** dla SVDM',
  - numeryczne: **1**.
- Dla atrybutow numerycznych odleglosc jest normalizowana przez zakres (max-min).
- W SVDM wagi atrybutow `w_i` przyjete jako **1.0** (brak doprecyzowanej procedury wagowania).
- Przy remisie wybor klasy rozstrzygany jest leksykograficznie po etykiecie klasy.
- Domyslne k: `1`, `3`, `log2(n)`.
- Dla k+NN: lokalna SVDM jest liczona na zbiorze **N(x,n)**, gdzie `n` domyslnie = liczba obiektow treningowych, ale zawsze `n >= k`.

## 3. Postac danych wejsciowych
Format ARFF (przyklad skrócony):
```
@RELATION example
@ATTRIBUTE attr1 numeric
@ATTRIBUTE attr2 {A,B,C}
@ATTRIBUTE class {yes,no}
@DATA
1.2,A,yes
?,B,no
```
Wazne cechy:
- Dane sa rozdzielane przecinkiem lub bialymi znakami.
- Linie komentarzy zaczynaja sie od `%` lub `#`.
- Ostatnia kolumna to klasa decyzyjna.

Pliki przykladowe znajduja sie w katalogu `data/`.

## 4. Postac danych wyjsciowych
Dla kazdego eksperymentu program tworzy osobny folder:
```
<outdir>/<nazwa_pliku_wejsciowego>/EXP_<suffix>/
```
Gdzie `<suffix>` zawiera m.in. nazwe algorytmu, liczbe atrybutow, liczbe obiektow, k, rodzaj SVDM i tryb (g/l).

W tym folderze zapisywane sa pliki:
1. **OUT_...csv**
   - `id, atrybuty..., RId, CId, NCId`
   - RId: klasa rzeczywista, CId: decyzja standardowa, NCId: decyzja znormalizowana.
2. **STAT_...txt**
   - podsumowanie eksperymentu: parametry, czasy, statystyki atrybutow, macierze SVDM,
     macierze pomylek, precision/recall/F1 (standard i znormalizowane).
3. **kNN_...csv**
   - lista k najblizszych sasiadow z odleglosciami dla kazdego obiektu.

## 5. Kwestie projektowe i implementacyjne
### 5.1 Podzial na moduly
- `src/main.cpp` – konfiguracja, uruchamianie eksperymentow, sterowanie przeplywem.
- `src/arff_reader.cpp` + `include/arff_reader.h` – parser ARFF.
- `src/distance.cpp` + `include/distance.h` – SVDM i dystanse.
- `src/algorithms.cpp` + `include/algorithms.h` – RIONA/RIA/k+NN oraz g-rule.
- `src/metrics.cpp` + `include/metrics.h` – macierze pomylek i miary.
- `src/output.cpp` + `include/output.h` – zapis wynikow.
- `src/util.cpp` + `include/util.h` – pomocnicze funkcje string/parsingu.
- `include/dataset.h` – struktury danych (Dataset, Instance, Stats itd.).

### 5.2 Diagram zaleznosci (uproszczony)
```
 main.cpp
   |-- ArffReader -> Dataset
   |-- Distance (SVDM, InstanceDistance)
   |-- Algorithms (RIONA/RIA/k+NN)
   |-- Metrics (confusion, precision/recall/F1)
   |-- Output (OUT/STAT/kNN)
```

### 5.3 Przeplyw programu
1) Wczytanie danych ARFF -> `Dataset`.
2) (Opcjonalnie) nadpisanie typow atrybutow `--types`.
3) Wyznaczenie globalnych statystyk (min/max/range, SVDM).
4) Klasyfikacja leave-one-out w trybie g/l.
5) Wyznaczenie wynikow (standard i znormalizowane) + metryk.
6) Zapis plikow OUT/STAT/kNN do folderu eksperymentu.

## 6. Podrecznik uzytkownika
### 6.1 Kompilacja (przyklad: Clang + MSVC toolchain)
Uruchom w **x64 Native Tools Command Prompt for VS 2019**:
```
"C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -O2 -Wall -Wextra ^
  src\main.cpp src\util.cpp src\arff_reader.cpp src\distance.cpp src\algorithms.cpp src\metrics.cpp src\output.cpp ^
  -I include -o riona.exe
```

### 6.2 Kompilacja z CMake
```
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Release
```

### 6.3 Uruchomienie
```
riona.exe --input data\heart-statlog.arff
```

Przyklady parametrow:
- `--algo riona|ria|knn|all`
- `--mode g|l|both`
- `--svdm svdm|svdmprime`
- `--k 1,3,log`
- `--n <int>` (dla k+NN)
- `--missing <token>`
- `--outdir <folder>`

Przyklad pelny:
```
riona.exe --input data\yeast-mini.arff --algo all --mode both --svdm svdm --k 1,3,log --outdir results
```

Wyniki znajdziesz w:
```
results\yeast-mini\EXP_<suffix>\
```

---
Koniec raportu.