
src_points = [-1.0, 0.4, 0.42, 0.44, 0.48, 0.53, 0.57, 1.0]
dst_points = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
src_points = [-1.0, 0.3, 0.35, 0.387, 0.396, 0.431, 0.465, 1.0]
dst_points = [0.0, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]

src_points = [-1.0, 0.3, 0.35, 0.387, 0.396, 0.431, 0.465, 1.0]
dst_points = [0.0, 0.3, 0.4, 0.5, 0.6, 0.85, 0.95, 1.0]

def similaritymap(sim):
    srcLen = len(src_points)
    if sim <= src_points[0]:
        return 0.0
    elif sim >= src_points[srcLen - 1]:
        return 1.0
    result = 0.0
    for i in range(1, srcLen):
        if sim < src_points[i]:
            result = dst_points[i - 1] + (sim - src_points[i - 1]) * (dst_points[i] - dst_points[i - 1]) / (
                    src_points[i] - src_points[i - 1])
            break
    return result