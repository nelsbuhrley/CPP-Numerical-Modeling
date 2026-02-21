#include <iostream>
#include <limits>

using namespace std;

long long collatzSequence(long long n, long long& maxNum) {
    maxNum = n;

    while (n != 1) {
        if (n > maxNum) {
            maxNum = n;
        }

        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
    }

    return maxNum;
}

int main() {
    long long current = 1;
    long long maxInSequence;
    char input;

    cout << "3n+1 Problem - Press SPACE to continue, 'q' to quit" << endl;
    cout << "============================================" << endl << endl;

    while (true) {
        cout << "Starting number: " << current << endl;
        long long count = 0;
        long long temp = current;
        while (temp != 1) {
            count++;
            if (temp % 2 == 0) {
                temp = temp / 2;
            } else {
                temp = 3 * temp + 1;
            }
        }
        count++;  // count the final 1
        cout << "Sequence length: " << count << endl;

        maxInSequence = collatzSequence(current, maxInSequence);

        cout << "Max number in sequence: " << maxInSequence << endl;

        cout << "Next starting number: " << (maxInSequence + 1) << endl;
        cout << "============================================" << endl;

        current = maxInSequence + 1;

        cout << "Press SPACE for next iteration or 'q' to quit: ";
        cin >> input;

        if (input == 'q' || input == 'Q') {
            break;
        }

        cout << endl;
    }

    return 0;
}