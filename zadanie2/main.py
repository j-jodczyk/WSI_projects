# eksperytmenty losowe odtwarzalne - seedy (dolaczyc do sprawozdania plik seeds.txt)
# warunek stopu: liczba iteracji, budzet wywolan funkcji celu 100*D
from matplotlib import pyplot as plt
from algorithm import f, q, evolution_strategy
from plotting import plot
#zbadaj zbieżność obu metod w zależności od rozmiaru
# populacji potomnej oraz początkowej wartości zasięgu mutacji
def main():

    x0 = [41.29, 16.8, 12.29, 47.18, 15.75, 11.3, 95.79, 87.4, 16.05, 7.87]
    x2 = [100]*10
    fig, axs = plt.subplots(3,3, sharex=True, sharey=True)
    plot([997, 998, 999, 1000, 1001], fig, axs, evolution_strategy, x2, f, [10, 50, 100], [1, 10, 0.1])

    plt.show()


if __name__=="__main__":
    main()