#include <iostream>
using namespace std;

int collatz(int n) {
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        steps++;
    }
    return steps;
}

int main() {
    int maxSteps = 0;
    int maxNumber = 0;

    for (int i = 1; i <= 1000000; i++) {
        int steps = collatz(i);
        if (i % 1000 == 0) {
            cout << "Number " << i << " requires " << steps << " steps" << endl;
        }

        if (steps > maxSteps) {
            maxSteps = steps;
            maxNumber = i;
        }
    }

    cout << "Number " << maxNumber << " took the longest with " << maxSteps << " steps" << endl;
    return 0;
}