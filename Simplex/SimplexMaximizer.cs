/*
 * Copyright (c) 2008 Sedat Kapanoglu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
using System;
using System.Collections.Generic;
using System.Linq;

namespace Simplex
{
    /// <summary>
    /// Harikulade simplex solver class
    /// </summary>
    public sealed class SimplexMaximizer
    {
        public const int SimplexPrecision = 4;
        public IList<double> Solution { get { return finalSolution; } }

        public SimplexMaximizer(double[][] inputMatrix)
        {
            if (inputMatrix == null)
            {
                throw new ArgumentNullException("inputMatrix");
            }
            this.inputRows = inputMatrix;
            this.numRows = inputRows.Length;
            if (numRows < 2)
            {
                throw new ArgumentException("Should have at least two rows");
            }
        }

        private double[][] inputRows;
        private double[] finalSolution;
        private int numRows;

        /// <summary>
        /// Trims numbers to a specified precision
        /// </summary>
        private static double normalize(double number)
        {
            double pow = Math.Pow(10, SimplexPrecision);
            return Math.Truncate((number * pow)) / pow;
        }

        /// <summary>
        /// Solve a simplex maximization problem using two-phase approach. This method cannot be called twice.
        /// </summary>
        /// <returns>True if feasible solution is found, false otherwise</returns>
        public bool Solve()
        {
            int numPhaseOneColumns = inputRows[0].Length + inputRows.Length - 1;
            double[][] phaseOneRows = buildPhaseOneRows(numPhaseOneColumns);
            phaseOneRowOperations(numPhaseOneColumns, phaseOneRows);
            var interimSolution = solvePhaseTwo(phaseOneRows);
            if (normalize(interimSolution[numPhaseOneColumns - 1]) != 0)
            {
                finalSolution = null;
                return false;
            }
            transferSolutionToInput(numPhaseOneColumns, phaseOneRows);
            phaseTwoRowOperations();
            this.finalSolution = solvePhaseTwo(inputRows);
            return true;
        }

        private static int findVariableLocation(double[][] rows, double[] row)
        {
            for (var column = 0; column < rows[0].Length - 1; column++)
            {
                if (normalize(row[column]) != 1)
                {
                    continue;
                }
                bool found = rows
                    .Where(otherRow => otherRow != row)
                    .Any(otherRow => normalize(otherRow[column]) != 0);
                if (!found)
                {
                    return column;
                }
            }
            throw new InvalidOperationException("Couldn't find variable location at row " + row.ToString());
        }

        private void phaseTwoRowOperations()
        {
            var firstRow = inputRows.First();
            foreach (var row in inputRows.Skip(1))
            {
                int column = findVariableLocation(inputRows, row);
                double factor = -firstRow[column];
                for (int x = 0; x < firstRow.Length; x++)
                {
                    firstRow[x] += row[x] * factor;
                }
            }
        }

        private void transferSolutionToInput(int numColumns, double[][] rows)
        {
            int rhsColumn = inputRows[0].Length - 1;
            for (int y = 1; y < numRows; y++)
            {
                Array.Copy(rows[y], inputRows[y], rhsColumn);
                inputRows[y][rhsColumn] = rows[y][numColumns - 1];
            }
        }

        private void phaseOneRowOperations(int numColumns, double[][] rows)
        {
            for (var column = 0; column < numColumns; column++)
            {
                double sum = 0;
                for (int row = 1; row < numRows; row++)
                {
                    sum += rows[row][column];
                }
                var firstRow = rows.First();
                firstRow[column] = firstRow[column] - sum;
            }
        }

        private double[][] buildPhaseOneRows(int numColumns)
        {
            var result = new double[numRows][];
            // set all R coefficients to 1 for the objective function
            var firstRow = new double[numColumns];
            for (int column = inputRows[0].Length - 1; column < numColumns - 1; column++)
            {
                firstRow[column] = 1;
            }
            result[0] = firstRow;
            for (int y = 1; y < numRows; y++)
            {
                var row = new double[numColumns];
                Array.Copy(inputRows[y], row, inputRows[y].Length);
                int rhsColumn = inputRows[0].Length - 1;
                row[numColumns - 1] = row[rhsColumn]; // move RHS to the end
                row[rhsColumn] = 0;
                // set appropriate R coefficient to 1
                row[rhsColumn + y - 1] = 1;
                result[y] = row;
            }
            return result;
        }

        /// <summary>
        /// Solve phase two of simplex problem
        /// </summary>
        /// <returns>true if a feasible solution is found, false otherwise</returns>
        private static double[] solvePhaseTwo(double[][] rows)
        {
            if (rows.Length < 2)
            {
                throw new InvalidOperationException("Must have at least 2 rows");
            }
            bool solutionFound = false;
            var firstRow = rows.First();
            int numColumns = firstRow.Length;
            int rhsColumn = numColumns - 1;
            while (!solutionFound)
            {
                int largestElementColumn = findLargestAbsoluteElement(firstRow, numColumns);
                var pivotRow = findPivotRow(rows, rhsColumn, largestElementColumn);
                processPivotRow(numColumns, largestElementColumn, pivotRow);
                applyPivotToRows(rows, largestElementColumn, pivotRow);
                solutionFound = isOptimal(firstRow);
            }
            var solution = new double[numColumns]; // Z is at the end
            solution[numColumns - 1] = firstRow[rhsColumn]; // Z
            foreach (var row in rows.Skip(1))
            {
                // find which variable RHS belongs to
                int location = findVariableLocation(rows, row);
                solution[location] = row[rhsColumn];
            }
            return solution;
        }

        private static void applyPivotToRows(double[][] rows, int largestElementColumn, double[] pivotRow)
        {
            foreach (var row in rows.Where(r => r != pivotRow))
            {
                double multiplier = -row[largestElementColumn];
                for (int column = 0; column < row.Length; column++)
                {
                    row[column] += (pivotRow[column] * multiplier);
                }
            }
        }

        private static void processPivotRow(int numColumns, int largestElementColumn, double[] pivotRow)
        {
            double pivotValue = pivotRow[largestElementColumn];
            for (int column = 0; column < numColumns; column++)
            {
                pivotRow[column] /= pivotValue;
            }
        }

        private static bool isOptimal(IList<double> row)
        {
            return !row.Take(row.Count - 1).Any(x => x < 0);
        }

        private static double[] findPivotRow(double[][] rows, int rhsColumn, int largestElementColumn)
        {
            double min = 0;
            double[] pivotRow = null;
            foreach (var row in rows.Where(row => row[largestElementColumn] > 0))
            {
                double factor = row[rhsColumn] / row[largestElementColumn];
                if (pivotRow == null || factor < min)
                {
                    min = factor;
                    pivotRow = row;
                }
            }
            if (pivotRow == null)
            {
                throw new InvalidOperationException("Could not find the pivot");
            }
            return pivotRow;
        }

        private static int findLargestAbsoluteElement(double[] row, int numColumns)
        {
            double max = 0;
            int result = -1;
            for (int column = 0; column < numColumns - 1; column++)
            {
                double value = row[column];
                if (value >= 0)
                {
                    continue;
                }
                if (result < 0 || -value > max)
                {
                    result = column;
                    max = -value;
                }
            }
            if (result < 0)
            {
                throw new InvalidOperationException("Could not find a max");
            }
            return result;
        }
    }
}
