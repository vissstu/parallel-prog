import numpy as np
import re
from pathlib import Path
from collections import defaultdict


class MatrixLoader:
    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Файл {file_path} пуст")
        n = int(lines[0])
        if len(lines) - 1 != n:
            raise ValueError(f"В файле {file_path} указан размер {n}, но найдено {len(lines) - 1} строк")
        matrix = []
        for i, line in enumerate(lines[1:], start=1):
            parts = line.split()
            if len(parts) != n:
                raise ValueError(f"Строка {i} файла {file_path} содержит {len(parts)} чисел, ожидалось {n}")
            matrix.append(list(map(float, parts)))
        return np.array(matrix, dtype=np.float64)


def validate_matrices(A, B, C, tol=1e-6):
    expected = np.dot(A, B)
    diff = np.abs(expected - C)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    success = max_diff <= tol
    if not success:
        max_pos = np.unravel_index(np.argmax(diff), diff.shape)
        return {
            'success': False,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'max_pos': max_pos,
            'expected': expected[max_pos],
            'computed': C[max_pos]
        }
    else:
        return {
            'success': True,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        }


def find_matrix_sets(directory='.'):
    """Находит все наборы файлов matrixA_*_*.txt, matrixB_*_*.txt, matrixC_*_*.txt."""
    pattern = re.compile(r'matrix([ABC])_(\d+)_(\d+)\.txt')
    sets = defaultdict(dict)
    for file in Path(directory).glob('matrix*.txt'):
        m = pattern.match(file.name)
        if m:
            letter = m.group(1)
            exp_num = int(m.group(2))
            size = int(m.group(3))
            key = (exp_num, size)
            sets[key][letter] = file
    complete = {k: v for k, v in sets.items() if len(v) == 3}
    return complete


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Проверка умножения матриц")
    parser.add_argument('--dir', default='.', help="Каталог с файлами (по умолчанию текущий)")
    parser.add_argument('--tol', type=float, default=1e-6, help="Допустимое отклонение")
    parser.add_argument('--report', '-r', help="Сохранить отчёт в файл")
    parser.add_argument('--single', nargs=3, metavar=('A', 'B', 'C'),
                        help="Проверить только один набор файлов (три файла)")
    args = parser.parse_args()

    if args.single:
        # Режим одного набора
        a_file, b_file, c_file = args.single
        try:
            A = MatrixLoader.load(a_file)
            B = MatrixLoader.load(b_file)
            C = MatrixLoader.load(c_file)
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return 1
        if A.shape[0] != B.shape[0] or A.shape[0] != C.shape[0]:
            print("Ошибка: размеры матриц не совпадают")
            return 1
        result = validate_matrices(A, B, C, args.tol)
        print_results(result, {'A': a_file, 'B': b_file, 'C': c_file}, args.tol)
        if args.report:
            save_report(result, args.single, args.tol, args.report)
    else:
        # Автоматический режим
        sets = find_matrix_sets(args.dir)
        if not sets:
            print(f"В каталоге {args.dir} не найдено полных наборов matrix[A,B,C]_*_*.txt")
            return 1
        results = []
        for (exp, size), files in sorted(sets.items()):
            try:
                A = MatrixLoader.load(files['A'])
                B = MatrixLoader.load(files['B'])
                C = MatrixLoader.load(files['C'])
                if A.shape[0] != size or B.shape[0] != size or C.shape[0] != size:
                    print(f"Предупреждение: размеры в файлах набора exp={exp}, size={size} не совпадают с именем")
                    continue
                result = validate_matrices(A, B, C, args.tol)
                result['exp'] = exp
                result['size'] = size
                result['files'] = {k: str(v) for k, v in files.items()}
                results.append(result)
            except Exception as e:
                print(f"Ошибка при обработке набора exp={exp}, size={size}: {e}")

        print_summary(results, args.tol)
        if args.report:
            save_full_report(results, args.tol, args.report)


def print_results(result, files, tol):
    n = None
    print("=" * 60)
    print("РЕЗУЛЬТАТ ПРОВЕРКИ")
    print("=" * 60)
    print(f"Файлы: {files['A']}, {files['B']}, {files['C']}")
    print(f"Допустимое отклонение: {tol}")
    print("-" * 60)
    if result['success']:
        print("СТАТУС: УСПЕШНО")
    else:
        print("СТАТУС: ОШИБКА")
    print(f"Макс. расхождение: {result['max_diff']:.6e}")
    print(f"Средн. расхождение: {result['mean_diff']:.6e}")
    if not result['success']:
        print(f"Позиция: {result['max_pos']}")
        print(f"Ожидаемое: {result['expected']:.6f}")
        print(f"Полученное: {result['computed']:.6f}")
    print("=" * 60)


def print_summary(results, tol):
    success_count = sum(1 for r in results if r['success'])
    total = len(results)
    print("=" * 70)
    print("СВОДНЫЙ ОТЧЁТ ПО ПРОВЕРКЕ УМНОЖЕНИЯ МАТРИЦ")
    print("=" * 70)
    print(f"Найдено наборов: {total}")
    print(f"Успешно: {success_count}")
    print(f"Ошибки: {total - success_count}")
    if success_count == total:
        print("ВСЕ РЕЗУЛЬТАТЫ ПРОШЛИ ПРОВЕРКУ!")
    else:
        print("ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ!")
    print("-" * 70)
    print(f"|{'Эксп.':^6}|{'Размер':^8}|{'Статус':^12}|{'Макс. расх.':^20}|{'Средн. расх.':^20}|")
    print("-" * 70)
    for r in results:
        status = "Успех" if r['success'] else "Ошибка"
        print(f"|{r['exp']:6d}|{r['size']:8d}|{status:^12}|{r['max_diff']:20.6e}|{r['mean_diff']:20.6e}|")
    print("-" * 70)


def save_report(result, files, tol, report_file):
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЁТ ПО ПРОВЕРКЕ УМНОЖЕНИЯ МАТРИЦ\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Файл A: {files[0]}\n")
        f.write(f"Файл B: {files[1]}\n")
        f.write(f"Файл C: {files[2]}\n")
        f.write(f"Допустимое отклонение: {tol}\n\n")
        f.write(f"Статус: {'УСПЕШНО' if result['success'] else 'ОШИБКА'}\n")
        f.write(f"Макс. расхождение: {result['max_diff']:.6e}\n")
        f.write(f"Средн. расхождение: {result['mean_diff']:.6e}\n")
        if not result['success']:
            f.write(f"Позиция максимального расхождения: {result['max_pos']}\n")
            f.write(f"Ожидаемое значение: {result['expected']:.6f}\n")
            f.write(f"Вычисленное значение: {result['computed']:.6f}\n")
    print(f"Отчёт сохранён в {report_file}")


def save_full_report(results, tol, report_file):
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("СВОДНЫЙ ОТЧЁТ ПО ПРОВЕРКЕ УМНОЖЕНИЯ МАТРИЦ\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Допустимое отклонение: {tol}\n\n")
        f.write("Результаты по каждому набору:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"Эксперимент {r['exp']}, размер {r['size']}x{r['size']}\n")
            f.write(f"  Файлы: {r['files']['A']}, {r['files']['B']}, {r['files']['C']}\n")
            f.write(f"  Статус: {'УСПЕШНО' if r['success'] else 'ОШИБКА'}\n")
            f.write(f"  Макс. расхождение: {r['max_diff']:.6e}\n")
            f.write(f"  Средн. расхождение: {r['mean_diff']:.6e}\n")
            if not r['success']:
                f.write(f"  Позиция ошибки: {r['max_pos']}\n")
                f.write(f"  Ожидаемое значение: {r['expected']:.6f}\n")
                f.write(f"  Вычисленное значение: {r['computed']:.6f}\n")
            f.write("\n")
        f.write("=" * 70 + "\n")
    print(f"Отчёт сохранён в {report_file}")


if __name__ == "__main__":
    main()