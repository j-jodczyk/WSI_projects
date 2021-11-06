# eksperytmenty losowe odtwarzalne - seedy (dolaczyc do sprawozdania plik seeds.txt)
# warunek stopu: liczba iteracji, budzet wywolan funkcji celu 100*D
import numpy as np
from algorithm import f, evolution_strategy
#zbadaj zbieżność obu metod w zależności od rozmiaru
# populacji potomnej oraz początkowej wartości zasięgu mutacji
def main():
    np.random.seed(9)
    x0 = np.random.random(10)*100
    print(evolution_strategy(x0, f, 10, 1, 'SA'))

if __name__=="__main__":
    main()