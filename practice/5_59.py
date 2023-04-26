# medium
# Spiral Matrix II  螺旋矩阵

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        startx, starty = 0, 0  # 起始位置
        indent, num = 1, 1  # 缩进，数字的初始化
        matrix = [[0] * n for _ in range(n)]
        for loop in range(n // 2):    # 分别对四条边进行填充
            for column in range(startx, n - indent):
                matrix[starty][column] = num
                num += 1
            for row in range(starty, n - indent):
                matrix[row][n - indent] = num
                num += 1
            for column in range(n - indent, startx, -1):
                matrix[n - indent][column] = num
                num += 1
            for row in range(n - indent, starty, -1):
                matrix[row][startx] = num
                num += 1
            indent += 1
            startx += 1
            starty += 1
        if n % 2 != 0:  # 如果n为奇数，需要单独填充矩阵最中间的数字
            matrix[startx][starty] = num
        return matrix
