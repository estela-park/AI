def solution(N, stages):
    for_stage = [sum(1 for i in stages if i == j) for j in range(1, N + 2)]
    map(lambda e: e[0], sorted({i+1 : for_stage[i] / (sum(for_stage) - sum(for_stage[j] for j in range(i))) if i else (for_stage[i] / sum(for_stage)) for i in range(N)}.items(), key=lambda x: x[1], reverse=True))