from pm4py.objects.log.obj import EventLog, Trace

def segment_log(log: EventLog, window_size: int = 200):
    """
    Разбивает log на окна по window_size traces.

    :param log: EventLog (pm4py EventLog)
    :param window_size: размер окна (по количеству traces)
    :return: список окон (каждое окно — список traces)
    """
    windows = []
    current_window = []

    for i, trace in enumerate(log):
        current_window.append(trace)

        if (i + 1) % window_size == 0:
            windows.append(current_window)
            current_window = []

    if current_window:
        windows.append(current_window)

    print(f"\nSegmented log into {len(windows)} windows (window size = {window_size} traces).")
    return windows

def segment_log_fixed_n(log, n_windows: int = 100):
    """
    Делит лог на n окон с равным количеством трейсов (последнее может быть короче).
    """
    trace_list = list(log)
    n_traces = len(trace_list)
    window_size = max(1, n_traces // n_windows)

    windows = []
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < n_windows - 1 else n_traces
        windows.append(trace_list[start:end])
    return windows