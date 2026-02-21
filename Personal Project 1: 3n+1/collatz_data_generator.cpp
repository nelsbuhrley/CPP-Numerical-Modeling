/*
 * Collatz (3n+1)+1 Sequence Data Generator
 *
 * Generates data for the first n max numbers in the sequence
 * Output format: index,max_value (CSV for easy plotting)
 *
 * Build and Run: Press F5
 */

#include <fstream>
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
    int numPoints;
    string filename;

    cout << "Collatz (3n+1)+1 Sequence Data Generator" << endl;
    cout << "========================================" << endl << endl;

    cout << "Enter number of max values to generate: ";
    cin >> numPoints;

    cout << "Enter output filename (e.g., collatz_data.csv): ";
    cin >> filename;

    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cerr << "Error: Could not open file for writing!" << endl;
        return 1;
    }

    // Write CSV header
    outFile << "index,max_value" << endl;

    long long current = 1;
    long long maxInSequence;

    cout << endl << "Generating data..." << endl;

    for (int i = 0; i < numPoints; i++) {
        maxInSequence = collatzSequence(current, maxInSequence);

        // Write to file: index, max_value
        outFile << i << "," << maxInSequence << endl;

        // Progress indicator
        if ((i + 1) % 100 == 0 || i == numPoints - 1) {
            cout << "  Generated " << (i + 1) << " / " << numPoints << " points" << endl;
        }

        // Next starting number is (max + 1) following the (3n+1)+1 pattern
        current = maxInSequence + 1;
    }

    outFile.close();

    cout << endl << "✓ Data successfully written to: " << filename << endl;
    cout << "✓ Total data points: " << numPoints << endl;
    cout << endl << "Ready to plot!" << endl;

    return 0;
}
