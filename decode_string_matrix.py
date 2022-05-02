#! /usr/bin/env python3
import numpy as np

m = 700
n = 500

class MatrixDecoder():
    def decoder(string):
        '''COMMA = np.uint8(ord(','))
        CR = np.uint8(ord('\r'))
        LF = np.uint8(ord('\n'))
        ZERO = np.uint8(ord('0'))

        # Initialization

        res = np.empty(m*n, dtype=np.float)

        # Fill the matrix

        curInt = 0
        curPos = 0
        lastCharIsDigit = True

        for i in range(len(string)):
            c = string[i]
            if c == CR or c == LF or c == COMMA:
                if lastCharIsDigit:
                    # Write the last int in the flatten matrix
                    res[curPos] = curInt
                    curPos += 1
                    curInt = 0
                lastCharIsDigit = False
            else:
                curInt = curInt * 10 + (c - ZERO)
                lastCharIsDigit = True

        return res.reshape(m, n)'''
        matrix = []
        string_list = string.split('\n')
        print(type(string_list))
        matrix = []
        string_list2 = []
        for i in range(1,len(string_list)-2):
            string_list2.append(string_list[i])
        digits = 3
        for i in range(len(string_list2)):
            if i==10:
                digits=4
            if i==100:
                digits=5
            string_list2[i] = string_list2[i][digits:len(string_list2[i])]
            matrix.append(string_list2[i].split())
        print(matrix)
        print(len(matrix[0]))
        
        return matrix
        


def main():
    print('here')

if __name__ == "__main__":
    main()