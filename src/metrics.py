# src/metrics.py

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from special4pm.estimation.metrics import hill_number, estimate_species_richness_chao



def compute_dfg(window_traces):
    """
    Строим DFG (directly-follows graph) для одного окна.
    :param window_traces: список traces (1 window)
    :return: DFG dict
    """
    from pm4py.objects.log.obj import EventLog
    event_log = EventLog(window_traces)
    dfg = dfg_discovery.apply(event_log)
    return dfg

def dfg_to_species_list(dfg):
    """
    Преобразуем DFG → species list (list of counts of directly-follows pairs).
    Это нужно для Chao1 и Hill.
    """
    species_counts = list(dfg.values())
    return species_counts

def compute_chao1(species_list):
    """
    Считаем Chao1 (species richness estimation).
    """
    species_counts = {i: count for i, count in enumerate(species_list)}
    return estimate_species_richness_chao(species_counts)

def compute_hill(species_list, q=1):
    """
    Считаем Hill number q.
    """
    species_counts = {i: count for i, count in enumerate(species_list)}
    return hill_number(q, species_counts)
