# RIONA (C++)

## Opis zadania
Masz zaimplementować algorytmy: (1) RIONA (oparty na k+NN i regułach), (2) RIA (wykorzystujący tylko reguły) oraz (3) k+NN. Jednym z parametrów programu ma być miara odległości na atrybutach nominalnych, którą ma być SVDM

SVDM (Selective Value Difference Metric) to metryka odległości dla atrybutów nominalnych w uczeniu maszynowym, która selekcjonuje podzbiór atrybutów przed obliczeniem odległości między instancjami, poprawiając wydajność algorytmów jak k-NN.

Funkcja Induce_SVDM(U: zbiór treningowy):
  Dla każdego atrybutu i:
    Oblicz P(dec=v | a_i=x) dla wszystkich v w Vd, x w wartościach a_i na U
    Oblicz wagi w_i (np. wg procedury z [8])
  Zwróć ρ: ρ(x,y) = sum_{i=1 do n} w_i * ρ_i(x_i, y_i)
    gdzie ρ_i(x_i, y_i) = sum_{v w Vd} |P(v|x_i) - P(v|y_i)|


- Algorytmy ma wyznaczać odległości dla obiektów (powiedzmy p i q) na atrybucie (powiedzmy a), którego wartość nie jest znana dla przynajmniej jednego z tych obiektów.
Jeśli a jest atrybutem nominalnym, to odległość SVDM między p i q ma być równe 2 (dla SVDM’ to odległość miałoby być wtedy równe 1). Jeśli a jest atrybutem numerycznym, to odległość między p i q ma być równe 1.

- Klasyfikacja obiektów zbioru danych ma być przeprowadzana w trybie „leave-one-out”, czyli dla każdego obiektu tego zbioru danych na podstawie wartości atrybutów pozostałych obiektów zbiorze. 

- Klasyfikacja obiektów zbioru danych powinna być poprzedzona wyznaczeniem wartości minimalnej, maksymalnej oraz zakresu (ang. range) dla każdego atrybutu numerycznego na podstawie wartości atrybutów całego zbioru danych oraz wyznaczeniem odległości odpowiednio SVDM lub SVDM’ pomiędzy wartościami atrybutów nominalnych.

- Eksperymenty należy przeprowadzać w 2 trybach wyznaczania odległości pomiędzy wartościami atrybutów obiektu aktualnie klasyfikowanego i innego: 
    1.	globalnym g – korzystającym z wyznaczonych podczas przetwarzania wstępnego na całym zbiorze danych (czyli włącznie z aktualnie klasyfikowanym obiektem) wartości min, maks, zakres dla atrybutów numerycznych i SVDM/SVDM’ dla atrybutów nominalnych,
    2.	lokalnym l – korzystającym z analogicznych wartości dla atrybutów, ale wyznaczanych na podstawie zbioru danych pomniejszonego o aktualnie klasyfikowany obiekt.
- Eksperymenty należy przeprowadzić dla różnych wartości parametru k, w tym dla k = 1, 3 i log2n, gdzie n jest liczbą obiektów w zbiorze.
- Należy zwracać dwie decyzje o przydzieleniu obiektu do klasy decyzyjnej:
    1)	decyzję standardową - wyznaczoną na podstawie liczby znalezionych obiektów wspierających dla każdej z klas decyzyjnych,
    2)	decyzję znormalizowaną - wyznaczoną na podstawie procentowej liczby znalezionych obiektów wspierających dla każdej z klas decyzyjnych, gdzie procentowa liczba znalezionych obiektów wspierających dla klasy decyzyjnej jest równa liczbie znalezionych obiektów wspierających, należących do klasy, przez liczność tej klasy.


ZWRACANE WYNIKI
Wyniki zwrócone przez algorytm grupowania dla danego zbioru danych i wartości parametrów należy zapisać w 3 plikach:

(1) OUT - plik wyjściowy zawierający w osobnej linii dla każdego obiektu zbioru następujące informacje:
     identyfikator obiektu, x, y, ...., RId, CId, NCId
gdzie:
- id obiektu - pozycja obiektu w zbiorze wejściowym,
- x, y, ... – wartości atrybutów (jeśli były zastępowane wartości nieznane, to należy podać wartości po zastąpieniu)
- RId – etykieta rzeczywistej klasy, do której obiekt należy,
- CId – etykieta wykrytej klasy, do której obiekt został przypisany na podstawie liczby znalezionych obiektów wspierających dla każdej z klas decyzyjnych.
- NCId – etykieta wykrytej klasy, do której obiekt został przypisany, na podstawie procentowej liczby znalezionych obiektów wspierających dla każdej z klas decyzyjnych.

(2) STAT - plik z następującymi informacjami:
- nazwa pliku wejściowego, liczba atrybutów obiektu, liczba obiektów w pliku wejściowym
- wartość parametru k
- miara odległości dla atrybutów nominalnych: odpowiednio SVDM lub SVDM’,
- częściowe czasy wykonania dla każdej istotnej fazy algorytmu jak m.in.: odczyt pliku wejściowego, operacje wykonane w trakcie przetwarzania wstępnego (np. obliczenie wartości min., maks. i zakres = maks. – min. dla atrybutów numerycznych; analogiczne obliczenia odległości pomiędzy wartościami atrybutów nominalnych), obliczenie k+NNs, …, zapisanie wyników do pliku wyjściowego,
- całkowity czas działania
- d = liczba klas decyzyjnych d
- liczności wszystkich klas decyzyjnych
- wartości min. maks. i zakres dla 1. atrybutu numerycznego,
…
- wartości min. maks. i zakres dla ostatniego atrybutu numerycznego,
- macierz odległości SVDM (lub SVDM’) pomiędzy wartościami 1. atrybutu nominalnego,
…
- macierz odległości SVDM (lub SVDM’) pomiędzy wartościami ostatniego atrybutu nominalnego,
- macierz konfuzji (confusion matrix) (Confusion matrix - Wikipedia)
- wartości następujących miar dla każdej i-tej klasy decyzyjnej przy przydzielaniu obiektów do klas decyzyjnych w sposób standardowy i znormalizowany: 
•	- Precyzja_i, NPrecyzja_i - # poprawnie sklasyfikowanych obiektów w i-tej klasie dec. do # wszystkich obiektów sklasyfikowanych jako należące do tej klasy,
•	- Odzysk_i, NOdzysk_i - # poprawnie sklasyfikowanych obiektów w i-tej klasie dec. do # wszystkich obiektów w tej klasie,
•	- F1_i, N F1_i - 2 * (Precyzja_i * Odzysk_i)/( Precyzja_i + Odzysk_i).
- wartości następujących miar dla całego zbioru danych przy przydzielaniu obiektów do klas decyzyjnych w sposób standardowy i znormalizowany::
•	- Bal_Precyzja, NBal_Precyzja - (i=1..d Precyzja_i)/d,
•	- Bal_Odzysk_i, NBal_Odzysk_i - (i=1..d Odzysk_i)/d
•	- Bal_F1_i, Bal_F1_i - (i=1..d F1_i)/d

(3) k+NN – plik zawierający następujące informacje w osobnej linii dla każdego obiektu p w zbiorze danych:
    id obiektu p, liczność k+NN(p), (id 1. obiektu w k+NN(p), odległość p do 1. obiektu w k+NN(p)), …, (id ostatniego obiektu w k+NN(p), odległość p do ostatniego obiektu w k+NN(p))


NAZWY PLIKÓW OUT, STAT i kNN
Nazwy plików OUT, STAT, k+NN powinny mieć związek z wykonywanym eksperymentem. Przykładowo nazwa pliku OUT przechowującego wyniki zwrócone przez wersję RIONA dla zbioru danych fname zawierającego 10000 4-atrybutowych obiektów i uruchomionego z parametrami k = 3, miarą SVDM’ i w trybie globalnym (g) prowadzenia klasyfikacji:
    OUT_RIONA_fname_D4_R10000_k3_SVDM’_g.csv
lub
    OUT_RIONA_fname_D4_R10000_k3_SVDM’_g.txt

## Pseudokody
k-NN
Algorytm Local_kNN(x: obiekt testowy, ρ globalna, k, n):
  N(x,n) = n najbliższych sąsiadów x z U wg ρ
  ρ_x = Induce_SVDM(N(x,n))  // lokalna SVDM
  S(x,k) = k najbliższych sąsiadów x z N(x,n) wg ρ_x
  Zwróć dec(x) = argmax_v |{y w S(x,k): dec(y)=v}|

RIA
Algorytm RIA(tst, trnSet, {ρ_a}_{a∈Asym}):
  Dla każdego v ∈ Vd:
    supportSet(v) = ∅
  Dla każdego trn ∈ trnSet:
    v = d(trn)
    Jeśli isCons(g-rule(tst, trn, {ρ_a}), trnSet):
      supportSet(v) = supportSet(v) ∪ {trn}
  Zwróć argmax_v |supportSet(v)|

RIONA
Algorytm RIONA(tst, trnSet, k_opt, ρ):  // k_opt z fazy uczenia
  N = N(tst, trnSet, k_opt, ρ)  // optymalne sąsiedztwo wg SVDM i euklides
  Zwróć RIA(tst, N, {ρ_a})     // RIA na sąsiedztwie zamiast całym trnSet
