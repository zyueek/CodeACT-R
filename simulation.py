import pandas as pd
import re
import argparse
import random
import os
import warnings
import statistics
import numpy as np
from tqdm import *
from path_seman import generate_path
from sti_with_backward import find_variable_and_statements_from_file
from collections import Counter
import math
import sys
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# pyACT-R Cognitive Architecture Integration for C++ Simulation
# ------------------------------------------------------------
PYACTR_DIR = Path(__file__).resolve().parent / "pyactr"
if str(PYACTR_DIR) not in sys.path:
    sys.path.insert(0, str(PYACTR_DIR))

from pyactr.chunks import chunktype, makechunk
PYACTR_AVAILABLE = True

# Chunk type definitions tailored for C++ comprehension tasks
chunktype("cpp_statement", "token line_num column semantic control feature complexity section stimulus pid")
chunktype("cpp_fixation", "line column semantic control feature complexity memtype attention_strength timestamp")
chunktype("cpp_attention", "timestamp focus_line cognitive_load working_memory_items inhibition_strength")
chunktype("cpp_memory_trace", "stimulus pid memtype step_index complexity activation last_timestamp")
chunktype("cpp_strategy", "strategy_name context applicability success_rate usage_count")
chunktype("cpp_goal_state", "goal_type current_focus target_line completion_status priority")
chunktype("cpp_step_context", "memtype step_index total_steps stimuli_count pid_count timestamp")
chunktype("cpp_retrieval_request", "memtype source_label candidate_count status timestamp")
chunktype("cpp_retrieval_result", "memtype line column complexity control feature status timestamp")
chunktype("cpp_action_execution", "memtype action_label line complexity status timestamp")


class CognitiveArchitecture:
    """
    Manages pyACT-R cognitive state for the overall C++ simulation without altering algorithms.
    """

    def __init__(self):
        self.declarative_memory = {}
        self.working_memory = []
        self.goal_stack = []
        self.fixation_history = []
        self.strategy_usage = {}
        self.attention_focus = None
        self.last_strategy_chunk = None
        self.step_history = []
        self.retrieval_requests = []
        self.retrieval_results = []
        self.actions = []
        self.buffers = {}
        self.cognitive_parameters = {
            'base_level_constant': 0.0,
            'activation_noise': 0.25,
            'retrieval_threshold': 0.0,
            'latency_factor': 0.1,
            'decay_rate': 0.5,
            'spreading_activation': 0.1,
            'associative_strength': 0.5,
            'goal_activation': 1.0,
            'working_memory_capacity': 7,
            'attention_span': 3.0,
            'fatigue_buildup_rate': 0.01,
        }
        self._initialize_semantic_knowledge()
        self.current_goal = makechunk(
            nameofchunk="cpp_initial_goal",
            typename="cpp_goal_state",
            goal_type="code_comprehension",
            current_focus=0,
            target_line=0,
            completion_status="active",
            priority=1.0,
        )

    @staticmethod
    def _safe_int(value):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _initialize_semantic_knowledge(self):
        constructs = [
            ("FunctionDefinition", 0.8, "high"),
            ("ForStmt", 0.7, "medium"),
            ("WhileStmt", 0.7, "medium"),
            ("IfStmt", 0.6, "high"),
            ("SwitchStmt", 0.6, "medium"),
            ("TryStmt", 0.5, "medium"),
            ("CatchStmt", 0.5, "medium"),
            ("ReturnStmt", 0.4, "low"),
            ("Assignment", 0.4, "low"),
            ("VariableDeclaration", 0.4, "low"),
        ]
        for construct, complexity_level, importance in constructs:
            knowledge_chunk = makechunk(
                nameofchunk=f"cpp_knowledge_{construct.lower()}",
                typename="cpp_statement",
                token=construct,
                line_num=-1,
                column=-1,
                semantic=construct,
                control=-1,
                feature=construct,
                complexity=complexity_level,
                section=0.0,
                stimulus=importance,
                pid="architect",
            )
            self.declarative_memory[construct] = knowledge_chunk

    def _calculate_attention_strength(self, complexity, section):
        complexity_value = self._safe_float(complexity) or 0.0
        section_value = self._safe_float(section)
        complexity_factor = min(2.0, 0.5 + complexity_value / 10.0)
        if section_value is not None:
            section_factor = max(0.5, min(1.5, 1.0 - abs(section_value - 0.5)))
        else:
            section_factor = 1.0
        return complexity_factor * section_factor

    def update_attention_state(self, line=None, complexity=None):
        line_num = self._safe_int(line)
        complexity_value = self._safe_float(complexity) or 0.0
        cognitive_load = min(1.0, complexity_value / 20.0)
        attention_chunk = makechunk(
            nameofchunk=f"cpp_attention_{len(self.working_memory)}",
            typename="cpp_attention",
            timestamp=datetime.now().isoformat(),
            focus_line=line_num if line_num is not None else -1,
            cognitive_load=cognitive_load,
            working_memory_items=len(self.working_memory),
            inhibition_strength=0.1,
        )
        self.working_memory.append(attention_chunk)
        if len(self.working_memory) > self.cognitive_parameters['working_memory_capacity']:
            self.working_memory.pop(0)
        self.attention_focus = attention_chunk

    def _store_memory_trace(self, stimuli, pid, memtype, complexity, timestamp):
        key = (str(stimuli or "unknown"), str(pid or "unknown"), str(memtype or "unknown"))
        complexity_value = self._safe_float(complexity) or 0.0
        activation = self.cognitive_parameters['base_level_constant'] + complexity_value * 0.05
        trace_chunk = makechunk(
            nameofchunk=f"cpp_memory_{len(self.declarative_memory)}",
            typename="cpp_memory_trace",
            stimulus=key[0],
            pid=key[1],
            memtype=key[2],
            step_index=len(self.fixation_history),
            complexity=complexity_value,
            activation=activation,
            last_timestamp=timestamp or datetime.now().isoformat(),
        )
        self.declarative_memory[key] = trace_chunk

    def register_strategy(self, memtype):
        label = str(memtype or "unknown")
        usage_count = self.strategy_usage.get(label, 0) + 1
        self.strategy_usage[label] = usage_count
        self.last_strategy_chunk = makechunk(
            nameofchunk=f"cpp_strategy_{label}_{usage_count}",
            typename="cpp_strategy",
            strategy_name=label,
            context="overall_simulation",
            applicability="cpp_navigation",
            success_rate=0.5,
            usage_count=usage_count,
        )

    def _set_buffer(self, buffer_name, chunk):
        self.buffers[buffer_name] = chunk

    def begin_step(self, memtype, step_index, total_steps, df_mem):
        try:
            stimuli_count = len(df_mem['stimuli'].unique())
        except Exception:
            stimuli_count = 0
        try:
            pid_count = len(df_mem['pid'].unique())
        except Exception:
            pid_count = 0
        step_chunk = makechunk(
            nameofchunk=f"cpp_step_{memtype}_{step_index}",
            typename="cpp_step_context",
            memtype=str(memtype or "unknown"),
            step_index=int(step_index),
            total_steps=int(total_steps),
            stimuli_count=int(stimuli_count),
            pid_count=int(pid_count),
            timestamp=datetime.now().isoformat(),
        )
        self.step_history.append(step_chunk)
        self._set_buffer('goal', step_chunk)
        return step_chunk

    def request_retrieval(self, memtype, source_label, candidate_count, success):
        status = "success" if success else "failure"
        request_chunk = makechunk(
            nameofchunk=f"cpp_retrieval_req_{memtype}_{len(self.retrieval_requests)}",
            typename="cpp_retrieval_request",
            memtype=str(memtype or "unknown"),
            source_label=str(source_label or "unknown"),
            candidate_count=int(candidate_count),
            status=status,
            timestamp=datetime.now().isoformat(),
        )
        self.retrieval_requests.append(request_chunk)
        self._set_buffer('retrieval_request', request_chunk)

    def record_retrieval_result(self, memtype, line=None, column=None, complexity=None,
                                control=None, feature=None, success=True):
        status = "success" if success else "failure"
        result_chunk = makechunk(
            nameofchunk=f"cpp_retrieval_res_{memtype}_{len(self.retrieval_results)}",
            typename="cpp_retrieval_result",
            memtype=str(memtype or "unknown"),
            line=self._safe_int(line) if line is not None else -1,
            column=self._safe_int(column) if column is not None else -1,
            complexity=self._safe_float(complexity) if complexity is not None else 0.0,
            control=self._safe_int(control) if control is not None else -1,
            feature=str(feature or "unknown"),
            status=status,
            timestamp=datetime.now().isoformat(),
        )
        self.retrieval_results.append(result_chunk)
        self._set_buffer('retrieval', result_chunk)
        return result_chunk

    def record_action(self, memtype, action_label, line=None, complexity=None, success=True):
        status = "success" if success else "failure"
        action_chunk = makechunk(
            nameofchunk=f"cpp_action_{memtype}_{len(self.actions)}",
            typename="cpp_action_execution",
            memtype=str(memtype or "unknown"),
            action_label=str(action_label or "unknown"),
            line=self._safe_int(line) if line is not None else -1,
            complexity=self._safe_float(complexity) if complexity is not None else 0.0,
            status=status,
            timestamp=datetime.now().isoformat(),
        )
        self.actions.append(action_chunk)
        self._set_buffer('manual', action_chunk)
        return action_chunk

    def register_statement(self, row):
        token = row.get('node') or row.get('token')
        line_num = self._safe_int(row.get('Line') or row.get('line'))
        column_num = self._safe_int(row.get('Column') or row.get('column'))
        semantic = row.get('semanic') or row.get('semantic') or "unknown"
        control = self._safe_int(row.get('controlnum'))
        feature = row.get('feature') or "unknown"
        complexity = self._safe_float(row.get('complexity')) or 0.0
        section = self._safe_float(row.get('section'))
        stimulus = row.get('stimuli') or row.get('node') or "unknown"
        pid = row.get('pid') or row.get('pidselect') or "unknown"
        statement_chunk = makechunk(
            nameofchunk=f"cpp_statement_L{line_num if line_num is not None else -1}_{len(self.goal_stack)}",
            typename="cpp_statement",
            token=str(token or "token"),
            line_num=line_num if line_num is not None else -1,
            column=column_num if column_num is not None else -1,
            semantic=str(semantic),
            control=control if control is not None else -1,
            feature=str(feature),
            complexity=complexity,
            section=section if section is not None else 0.0,
            stimulus=str(stimulus),
            pid=str(pid),
        )
        self.goal_stack.append(statement_chunk)
        if len(self.goal_stack) > self.cognitive_parameters['working_memory_capacity']:
            self.goal_stack.pop(0)
        self._set_buffer('imaginal', statement_chunk)
        return statement_chunk

    def record_fixation(self, line=None, column=None, semantic=None, control=None, feature=None,
                        complexity=None, memtype=None, section=None, stimuli=None, pid=None,
                        timestamp=None):
        line_num = self._safe_int(line)
        column_num = self._safe_int(column)
        control_num = self._safe_int(control)
        complexity_value = self._safe_float(complexity) or 0.0
        attention_strength = self._calculate_attention_strength(complexity_value, section)
        fixation_chunk = makechunk(
            nameofchunk=f"cpp_fixation_{len(self.fixation_history)}",
            typename="cpp_fixation",
            line=line_num if line_num is not None else -1,
            column=column_num if column_num is not None else -1,
            semantic=str(semantic or "unknown"),
            control=control_num if control_num is not None else -1,
            feature=str(feature or "unknown"),
            complexity=complexity_value,
            memtype=str(memtype or "unknown"),
            attention_strength=attention_strength,
            timestamp=timestamp or datetime.now().isoformat(),
        )
        self.fixation_history.append(fixation_chunk)
        self.update_attention_state(line=line, complexity=complexity_value)
        self._store_memory_trace(stimuli, pid, memtype, complexity_value, timestamp)
        self.register_strategy(memtype)


cognitive_architecture = CognitiveArchitecture()


def record_cognitive_fixation(df_result):
    """
    Capture the latest fixation into the pyACT-R cognitive architecture.
    """
    if not PYACTR_AVAILABLE or df_result is None or df_result.empty:
        return
    try:
        latest = df_result.iloc[-1]
    except Exception:
        return
    try:
        row_dict = latest.to_dict()
        timestamp = datetime.now().isoformat()
        cognitive_architecture.register_statement(row_dict)
        cognitive_architecture.record_fixation(
            line=row_dict.get('Line') or row_dict.get('line'),
            column=row_dict.get('Column') or row_dict.get('column'),
            semantic=row_dict.get('semanic') or row_dict.get('semantic'),
            control=row_dict.get('controlnum'),
            feature=row_dict.get('feature'),
            complexity=row_dict.get('complexity'),
            memtype=row_dict.get('memtype'),
            section=row_dict.get('section') or row_dict.get('corr_section'),
            stimuli=row_dict.get('stimuli') or row_dict.get('node'),
            pid=row_dict.get('pid') or row_dict.get('pidselect'),
            timestamp=timestamp,
        )
    except Exception:
        # Cognitive logging should never break the primary simulation flow.
        return

def extend_unique_values_with_tolerance(stepall, tolerance,discount):
    # Sort the values to cluster them by proximity
    sorted_values = sorted(stepall)

    # Initialize clusters
    clusters = []
    current_cluster = [sorted_values[0]]

    # Form clusters based on tolerance
    for value in sorted_values[1:]:
        if value <= current_cluster[-1] * (1 + tolerance):
            current_cluster.append(value)
        else:
            clusters.append(current_cluster)
            current_cluster = [value]
    clusters.append(current_cluster)

    # Count occurrences within each cluster
    unique_values = []
    for cluster in clusters:
        counts = Counter(cluster)
        for number, count in counts.items():
            if count > 3:
                unique_values.extend([number] * (count // discount))
            else:
                unique_values.extend([number] * count)

    return unique_values

def set_step_aug(df,step_length,evaselect):
# find the distribution of evaluation pid scan length to set the simulation step
  avglength=sum(len(sublist) for sublist in evaselect) / len(evaselect)
  dev=sum([abs(len(sublist) - avglength) for sublist in evaselect])/len(evaselect)/2
#  print(avglength,dev)
  if len(evaselect)==1:
    avglength=len(evaselect[0])
    dev=0.5*len(evaselect[0])
  stepall=step_length
#  step=random.choice(random.choice(stepall))
  chosen_value = random.gauss(avglength, dev)
  closest_val = min(stepall, key=lambda x: abs(x - chosen_value))
#  print(closest_val)
#  print(stepall)
  return closest_val
def set_step_zero(df,train_complexity,test_complexity):
  stepall=[]
  stepsti=[]
  complex_map = {}
  for sti in list(set(df['stimuli'].to_list())):
    df_sti=df[df['stimuli']==sti]
    for s in list(set(df_sti['pid'].to_list())):
      df_pid=df_sti[df_sti['pid']==s]
      if(len(df_pid)<50):
#      df_pid['dense_rank'] = df_pid['complexity'].rank(method='dense').astype(int)
#      print(df_pid)
#      print(df_pid['complexity'].rank(method='dense'))
        complex_map[tuple(df_pid['complexity'].rank(method='dense').tolist())]=df_pid['line'].to_list()
#      stepsti.append(df_pid['line'].to_list())
        stepall.append(len(df_pid)) 
  counts = Counter(stepall)
#  unique_values = []
  comp_measure=test_complexity/ (train_complexity + test_complexity) 
  tolerance=0
  discount=int(round(comp_measure*10,0))
  unique_values=extend_unique_values_with_tolerance(stepall,tolerance,discount)
  unique_values=stepall
  unique_values.sort() 
#  print(unique_values)
  central_index =  comp_measure * len(unique_values)
  random_index = int(np.round(np.random.normal(loc=central_index, scale=len(unique_values)//4)))
  random_index = max(0, min(random_index, len(unique_values) - 1))
  selected_value = unique_values[random_index]
#  step=random.choice(random.choice(stepall))
#  print(stepall)
  comp_list=[list(key) for key, value in complex_map.items() if len(value) == selected_value]
#  print(unique_values)
  return selected_value,comp_list,comp_measure
def path_match(df_result,next_counts,df_t):
  feature=df_result['feature'].iloc[-1]
  controlnum=df_result['controlnum'].iloc[-1]
  complexity=df_result['complexity'].iloc[-1]
#  mix=str(controlnum)+"_"+feature+"_"+str(complexity)
  mix=str(controlnum)+"_"+feature
  if mix in next_counts.keys():
    next=next_counts[mix]
    keys = list(next.keys())
    weights = list(next.values())
    selected_key = random.choices(keys, weights=weights, k=1)[0]
#    print(selected_key)
    target_rows= df_t[(df_t['controlnum'] == int(selected_key.split("_")[0])) & (df_t['feature'] == selected_key.split("_")[1])]
#    print(target_rows)
    return target_rows,mix
  else:
    return pd.DataFrame(),0
def path_match_line(df_result,line_next,df_t):
  line=str(df_result['Line'].iloc[-1])
#  print(line,line_next.keys())
  if line in line_next.keys():
    next=line_next[line]
    keys = list(next.keys())
    weights = list(next.values())
    selected_key = int(random.choices(keys, weights=weights, k=1)[0])
#    print(selected_key)
#    print(selected_key)
    target_rows= df_t[df_t['Line'] == int(selected_key)] 
#    print(target_rows)
    return target_rows ,line
  else:
    return pd.DataFrame(),0
def select_chunks(lst, num_chunks):
    random.shuffle(lst)
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks
def select_chunks_re(lst, chunk_percent):
  chunks = []
  for _ in range(10):
    # Calculate 30% of the list length
    chunk_size = max(1, int(len(lst) * (chunk_percent/10)))  # Ensure at least one element is selected
    # Randomly select a chunk of 30% of the list
    chunk = random.sample(lst, chunk_size)
    chunks.append(chunk)
  return chunks
def select_chunks_half(lst, aug,seed):
    random.seed(seed)

    half_chunk_size = len(lst) // 2  # Use integer division for exact half

    half_chunk = random.sample(lst, half_chunk_size)

    remaining_elements = list(set(lst) - set(half_chunk))

    percentage = int(aug[0])
    if (aug[0]=='9'):
        percentage=int(aug[0])+1
    chunk_size = max(1, int(len(remaining_elements) * percentage / 10))  

    chunk = random.sample(remaining_elements, chunk_size)
    return [(half_chunk, chunk)]
def stm(pidselect):
  df_stm=pd.DataFrame(columns=column)
  for pid in pidselect:
    df_pid=df_e[df_e['pid']==pid]
    df_stm=pd.concat([df_stm,df_pid])
    df_stm.reset_index(drop=True, inplace=True)
  return df_stm
def format_chunk_list(values):
  if values is None or values == 'null':
    return 'NA'
  if isinstance(values, str):
    return values
  try:
    return '_'.join(str(v) for v in values)
  except TypeError:
    return str(values)

def simulate(df_mem,step,df_result,memtype,pidselect,next_counts,short_next,lines,step_comp,line_next,stepall,half_chunk=None):
  half_chunk_repr = format_chunk_list(half_chunk)
  cognitive_architecture.begin_step(memtype, step, stepall, df_mem)
  if(memtype=='long_re'):
    timestamp=step/float(stepall)    
    selected_rows = df_mem[(df_mem['section'] > timestamp*0.9) & (df_mem['section'] < timestamp*1.1)]
    cognitive_architecture.request_retrieval(memtype, 'long_re_window', len(selected_rows), len(selected_rows) > 0)
    selected_rows['distance'] = abs(selected_rows['section'] - timestamp)
    selected_rows['weight'] = 1 / (selected_rows['distance'] + 1)
    selected_rows['weight'] = selected_rows['weight'] / selected_rows['weight'].sum()
    if(len(selected_rows)==0):
      print("no timestamp: "+step+timestep)
    random_row = selected_rows.sample(n=1, weights='weight')
    controlnum=random_row.iloc[0,4]
    feature=random_row.iloc[0,5]
    complexity=random_row.iloc[0,6]
    corr_section=random_row.iloc[0,7]
    target_rows_df = df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]
    if len(target_rows_df)==0:
      closest_index = (df_t['complexity'] - complexity).abs().idxmin()
      selected_series = df_t.loc[closest_index]
    elif len(target_rows_df)>1:
      closest_index = target_rows_df['complexity'].idxmax()
      selected_series = target_rows_df.loc[closest_index]
    else:
      selected_series = target_rows_df.iloc[0]
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows = selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+[corr_section,timestamp,memtype,'_'.join([str(p) for p in pidselect])]
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'long_re_selection',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
  if(memtype=='long'):
    timestamp=step/float(stepall)  
    selected_rows = df_mem[(df_mem['section'] > timestamp-0.1) & (df_mem['section'] < timestamp+0.1)]
    cognitive_architecture.request_retrieval(memtype, 'long_focus_window', len(selected_rows), len(selected_rows) > 0)
    df_t1['combined'] = list(zip(df_t1['controlnum'], df_t1['feature']))
    selected_rows['combined'] = list(zip(selected_rows['controlnum'], selected_rows['feature']))
    result = selected_rows[selected_rows['combined'].isin(df_t1['combined'])]
    cognitive_architecture.request_retrieval(memtype, 'long_matching', len(result), len(result) > 0)
    random_row = result.sample(n=1,random_state=int(args.seed))    
    controlnum=random_row.iloc[0,4]
    feature=random_row.iloc[0,5]
    complexity=random_row.iloc[0,6]
    corr_section=random_row.iloc[0,7]
    target_rows_df= df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]
    if len(target_rows_df)==0:
      closest_index = (df_t['complexity'] - complexity).abs().idxmin()
      selected_series = df_t.loc[closest_index]
    elif len(target_rows_df)>1:
      closest_index = target_rows_df['complexity'].idxmax()
      selected_series = target_rows_df.loc[closest_index]
      print(closest_index)
    else:
      selected_series = target_rows_df.iloc[0]
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows = selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+[corr_section,timestamp,memtype,'_'.join([str(p) for p in pidselect])]
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'long_selection',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
  if(memtype=='null'):
    df_t1 = df_t.copy(deep=True)
    timestamp=step/float(stepall)    
    selected_rows = df_mem[(df_mem['section'] > timestamp-0.1) & (df_mem['section'] < timestamp+0.1)]
    cognitive_architecture.request_retrieval(memtype, 'null_window', len(selected_rows), len(selected_rows) > 0)
    df_t1['combined'] = list(zip(df_t1['controlnum'], df_t1['feature']))
    selected_rows['combined'] = list(zip(selected_rows['controlnum'], selected_rows['feature']))
#    print(selected_rows)
#    print(df_t)
# Check if the combined tuples in df1 exist in df2
    result = selected_rows[selected_rows['combined'].isin(df_t1['combined'])]
    cognitive_architecture.request_retrieval(memtype, 'null_matching', len(result), len(result) > 0)
#    print(result)
#    selected_rows['distance'] = abs(selected_rows['section'] - timestamp)
#    selected_rows['weight'] = 1 / (selected_rows['distance'] + 1)
#    selected_rows['weight'] = selected_rows['weight'] / selected_rows['weight'].sum()

    if(len(selected_rows)==0):
      print("no timestamp: "+step+timestep)
#    random_row = selected_rows.sample(n=1, weights='weight',random_state=int(args.seed))
    random_row = result.sample(n=1,random_state=int(args.seed))
#    print(random_row)
    controlnum=random_row['controlnum'].iloc[0]
    feature=random_row['feature'].iloc[0]
    complexity=random_row['complexity'].iloc[0]
    corr_section=random_row['section'].iloc[0]
    target_rows_df= df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]
    if len(target_rows_df)>1:
      closest_index = target_rows_df['complexity'].idxmax()
      selected_series = target_rows_df.loc[closest_index]
    else:
      selected_series=target_rows_df.iloc[0,:]
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows=selected_series.to_list()
#    print(target_rows)
#    if len(target_rows)==0:
#      closest_index = (df_t['complexity'] - complexity).abs().idxmin()
#      target_rows = df_t.loc[closest_index].to_list()
#    elif len(target_rows)>1:
#      closest_index = target_rows['complexity'].idxmax()
#      target_rows = target_rows.loc[closest_index].to_list()
#    if(len(df_result)>=1 and target_rows[1]==df_result.iloc[-1,1]):
#    else:
#    print(target_rows)
    df_result.loc[len(df_result.index)]=target_rows+[corr_section,timestamp,memtype,'null']
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'null_selection',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
  if(memtype=='short_path'):
    timestamp=step/float(stepall)    
#    print(timestamp)
    if len(df_result)==0:  
      df_t1 = df_t.copy(deep=True)   
      selected_rows = df_mem[(df_mem['section'] > timestamp-0.1) & (df_mem['section'] < timestamp+0.1)]
      cognitive_architecture.request_retrieval(memtype, 'short_path_window', len(selected_rows), len(selected_rows) > 0)
      df_t1['combined'] = list(zip(df_t1['controlnum'], df_t1['feature']))
      selected_rows['combined'] = list(zip(selected_rows['controlnum'], selected_rows['feature']))
      result = selected_rows[selected_rows['combined'].isin(df_t1['combined'])]
      cognitive_architecture.request_retrieval(memtype, 'short_path_matching', len(result), len(result) > 0)
      if len(result)==0:
        cognitive_architecture.record_retrieval_result(memtype, success=False)
        cognitive_architecture.record_action(memtype, 'short_path_init', success=False)
        return False  
      random_row = result.sample(n=1,random_state=int(args.seed))
      controlnum=random_row['controlnum'].iloc[0]
      feature=random_row['feature'].iloc[0]
      target_rows_df= df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]
    else:
#      print(next_counts)
      target_rows_df,line=path_match_line(df_result,line_next,df_t)
      cognitive_architecture.request_retrieval(memtype, 'short_path_transition', len(target_rows_df), len(target_rows_df) > 0)
#      target_rows,line=path_match(df_result,short_next,df_t)
    if len(target_rows_df)==0:
      cognitive_architecture.record_retrieval_result(memtype, success=False)
      cognitive_architecture.record_action(memtype, 'short_path', success=False)
      return False
    elif len(target_rows_df)>=1:
      closest_index = target_rows_df['complexity'].idxmax()
      selected_series = target_rows_df.loc[closest_index]
#      print(closest_index)
#    print(target_rows)
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows = selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'_'.join([str(p) for p in pidselect]),half_chunk_repr]
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'short_path',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
    return True
  if(memtype=='long_path'):
    timestamp=step/float(stepall)    
    selected_rows = df_mem[(df_mem['section'] > timestamp-0.1) & (df_mem['section'] < timestamp+0.1)]
    cognitive_architecture.request_retrieval(memtype, 'long_path_window', len(selected_rows), len(selected_rows) > 0)
    selected_rows['distance'] = abs(selected_rows['section'] - timestamp)
    selected_rows['weight'] = 1 / (selected_rows['distance'] + 1)
    selected_rows['weight'] = selected_rows['weight'] / selected_rows['weight'].sum()
    random_row = selected_rows.sample(n=1, weights='weight')
    controlnum=random_row.iloc[0,4]
    feature=random_row.iloc[0,5]
    complexity=random_row.iloc[0,6]
    corr_section=random_row.iloc[0,7]
    target_rows_df= df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]    
    if (len(target_rows_df)==0 or len(target_rows_df)>1) and len(df_result)>1:
#      print(next_counts)
      alt_rows,mix=path_match(df_result,next_counts,df_t)
      if isinstance(alt_rows, pd.DataFrame):
        target_rows_df = alt_rows
      else:
        target_rows_df = pd.DataFrame(alt_rows)
      cognitive_architecture.request_retrieval(memtype, 'long_path_transition', len(target_rows_df), len(target_rows_df) > 0)
    if len(target_rows_df)==0:
      closest_index = (df_t['complexity'] - complexity).abs().idxmin()
      selected_series = df_t.loc[closest_index]
    elif len(target_rows_df)>1:
      closest_index = target_rows_df['complexity'].idxmax()
      selected_series = target_rows_df.loc[closest_index]
    else:
      selected_series = target_rows_df.iloc[0]
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows = selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+[corr_section,timestamp,memtype,'null']
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'long_path',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
  if(memtype=='short_path1'):
    timestamp=step/float(stepall)    
#    print(timestamp)
    if len(df_result)==0:
      selected_rows = df_mem[(df_mem['section'] > timestamp-0.1) & (df_mem['section'] < timestamp+0.1)]
      cognitive_architecture.request_retrieval(memtype, 'short_path1_window', len(selected_rows), len(selected_rows) > 0)
      if len(selected_rows)==0:
        cognitive_architecture.record_retrieval_result(memtype, success=False)
        cognitive_architecture.record_action(memtype, 'short_path1_init', success=False)
        return False
      selected_rows['distance'] = abs(selected_rows['section'] - timestamp)
      selected_rows['weight'] = 1 / (selected_rows['distance'] + 1)
      selected_rows['weight'] = selected_rows['weight'] / selected_rows['weight'].sum()
      random_row = selected_rows.sample(n=1, weights='weight')
      controlnum=random_row.iloc[0,4]
      feature=random_row.iloc[0,5]
      complexity=random_row.iloc[0,6]
      corr_section=random_row.iloc[0,7]
      target_rows= df_t[(df_t['controlnum'] == controlnum) & (df_t['feature'] == feature)]
    else:
#      print(next_counts)
      target_rows,line=path_match_line(df_result,line_next,df_t)
      cognitive_architecture.request_retrieval(memtype, 'short_path1_transition', len(target_rows), len(target_rows) > 0)
#      target_rows,line=path_match(df_result,short_next,df_t)
    if len(target_rows)==0:
#      print(i,line)
      cognitive_architecture.record_retrieval_result(memtype, success=False)
      cognitive_architecture.record_action(memtype, 'short_path1', success=False)
      return False
    elif len(target_rows)>1:
      closest_index = target_rows['complexity'].idxmax()
      selected_series = target_rows.loc[closest_index]
    else:
      selected_series = target_rows.iloc[0]
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    target_rows = selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'_'.join([str(p) for p in pidselect])]
    record_cognitive_fixation(df_result)
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'short_path1',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
    return True
  if(memtype=='quick'):
    timestamp=step/float(stepall)    
    variable, return_line, statements = find_variable_and_statements_from_file(lines)
    cognitive_architecture.request_retrieval(memtype, 'quick_statements', len(statements), len(statements) > 0)
#    print(statements)
#    print(variable, return_line, statements,i)
    if variable and (i<len(statements)-1):
      result_len=len(statements)
      target_line=statements[(-i)][1]
#      print(target_line)
      selected_series=df_t[(df_t['Line'] == target_line)].iloc[0]
      target_rows=selected_series.to_list()
#      print(target_rows,'asdasd')
#      print(df_t[(df_t['Line'] == target_line)].iloc[0])
      df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'null']
      record_cognitive_fixation(df_result)
      line_value = selected_series.get('Line', selected_series.get('line', -1))
      column_value = selected_series.get('Column', selected_series.get('column', -1))
      control_value = selected_series.get('controlnum')
      feature_value = selected_series.get('feature')
      complexity_value = selected_series.get('complexity')
      cognitive_architecture.record_retrieval_result(
          memtype,
          line=line_value,
          column=column_value,
          complexity=complexity_value,
          control=control_value,
          feature=feature_value,
          success=True
      )
      cognitive_architecture.record_action(
          memtype,
          'quick_strategy',
          line=line_value,
          complexity=complexity_value,
          success=True
      )
    else:
      closest_index = df_t['complexity'].idxmax()
      selected_series=df_t.loc[closest_index]
      target_rows=selected_series.to_list()
#      if((len(df_result)>0 and target_rows[1]!=df_result.iloc[-1,1]) or len(df_result)==0):
      df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'null']      
      record_cognitive_fixation(df_result)
      line_value = selected_series.get('Line', selected_series.get('line', -1))
      column_value = selected_series.get('Column', selected_series.get('column', -1))
      control_value = selected_series.get('controlnum')
      feature_value = selected_series.get('feature')
      complexity_value = selected_series.get('complexity')
      cognitive_architecture.record_retrieval_result(
          memtype,
          line=line_value,
          column=column_value,
          complexity=complexity_value,
          control=control_value,
          feature=feature_value,
          success=True
      )
      cognitive_architecture.record_action(
          memtype,
          'quick_fallback',
          line=line_value,
          complexity=complexity_value,
          success=True
      )
  if(memtype=='quick_comp'):
    timestamp=step/float(stepall)    
    complex_now=step_comp[i-1]
#    print(complex_now)
    df_t1=pd.read_csv(t+'_sem.csv')
    df_t1['rank'] = df_t1['complexity'].rank(method='dense').astype(int)
    df_t1 = df_t1.drop_duplicates(subset='Line', keep='first')
    total_rank = df_t1['rank'].sum()
    df_t1['probability'] = df_t1['rank'] / total_rank
    cognitive_architecture.request_retrieval(memtype, 'quick_comp_distribution', len(df_t1), len(df_t1) > 0)
    selected_row = df_t1.sample(n=1, weights='probability')
    target_rows = selected_row.drop(columns=['rank','probability'])
    selected_series=target_rows.iloc[0]
    target_rows=selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'null']   
    record_cognitive_fixation(df_result)
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'quick_comp',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
  if(memtype=='quick_aug'):
    timestamp=i/float(step)    
    variable, return_line, statements = find_variable_and_statements_from_file(lines)
    cognitive_architecture.request_retrieval(memtype, 'quick_aug_statements', len(statements), len(statements) > 0)
#    print(statements)
    if variable and (i<len(statements)-1):
      result_len=len(statements)
      target_line=statements[(-i-1)][1]
#      print(target_line)
      try:
        selected_series=df_t[(df_t['Line'] == target_line)].iloc[0,:]
      except:
        closest_index = df_t['complexity'].idxmax()
        selected_series=df_t.loc[closest_index]
      target_rows=selected_series.to_list()
      df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'_'.join([str(p) for p in pidselect]),half_chunk_repr]
      record_cognitive_fixation(df_result)
      line_value = selected_series.get('Line', selected_series.get('line', -1))
      column_value = selected_series.get('Column', selected_series.get('column', -1))
      control_value = selected_series.get('controlnum')
      feature_value = selected_series.get('feature')
      complexity_value = selected_series.get('complexity')
      cognitive_architecture.record_retrieval_result(
          memtype,
          line=line_value,
          column=column_value,
          complexity=complexity_value,
          control=control_value,
          feature=feature_value,
          success=True
      )
      cognitive_architecture.record_action(
          memtype,
          'quick_aug_strategy',
          line=line_value,
          complexity=complexity_value,
          success=True
      )
    else:
      closest_index = df_t['complexity'].idxmax()
      selected_series=df_t.loc[closest_index]
      target_rows=selected_series.to_list()
#      if(len(df_result)==0 or target_rows[1]!=df_result.iloc[-1,1]  or len(df_result)==0):
      df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'_'.join([str(p) for p in pidselect]),half_chunk_repr]   
      record_cognitive_fixation(df_result)
      line_value = selected_series.get('Line', selected_series.get('line', -1))
      column_value = selected_series.get('Column', selected_series.get('column', -1))
      control_value = selected_series.get('controlnum')
      feature_value = selected_series.get('feature')
      complexity_value = selected_series.get('complexity')
      cognitive_architecture.record_retrieval_result(
          memtype,
          line=line_value,
          column=column_value,
          complexity=complexity_value,
          control=control_value,
          feature=feature_value,
          success=True
      )
      cognitive_architecture.record_action(
          memtype,
          'quick_aug_fallback',
          line=line_value,
          complexity=complexity_value,
          success=True
      )
#      print(lines)
  if(memtype=='quick_aug_comp'):
    timestamp=step/float(stepall)    
#    complex_now=step_comp[i-1]
#    print(complex_now)
    df_t1=pd.read_csv(t+'_sem.csv')
    df_t1['rank'] = df_t1['complexity'].rank(method='dense').astype(int)
    df_t1 = df_t1.drop_duplicates(subset='Line', keep='first')
    total_rank = df_t1['rank'].sum()
    df_t1['probability'] = df_t1['rank'] / total_rank
    cognitive_architecture.request_retrieval(memtype, 'quick_aug_comp_distribution', len(df_t1), len(df_t1) > 0)
    selected_row = df_t1.sample(n=1, weights='probability')
    target_rows = selected_row.drop(columns=['rank','probability'])
    selected_series=target_rows.iloc[0]
    target_rows=selected_series.to_list()
    df_result.loc[len(df_result.index)]=target_rows+['0',timestamp,memtype,'_'.join([str(p) for p in pidselect]),half_chunk_repr] 
    record_cognitive_fixation(df_result)
    line_value = selected_series.get('Line', selected_series.get('line', -1))
    column_value = selected_series.get('Column', selected_series.get('column', -1))
    control_value = selected_series.get('controlnum')
    feature_value = selected_series.get('feature')
    complexity_value = selected_series.get('complexity')
    cognitive_architecture.record_retrieval_result(
        memtype,
        line=line_value,
        column=column_value,
        complexity=complexity_value,
        control=control_value,
        feature=feature_value,
        success=True
    )
    cognitive_architecture.record_action(
        memtype,
        'quick_aug_comp',
        line=line_value,
        complexity=complexity_value,
        success=True
    )
def train_complexity(slist):
  train_complexity=0
  for stim in slist:
    df_sem=pd.read_csv("stim_info/"+stim+'_sem.csv')
    unique_df = df_sem.drop_duplicates(subset='Line')
    total_sum = unique_df['complexity'].sum()
    train_complexity+=total_sum
#  print(train_complexity)
  return round(train_complexity/len(slist),2)
def action1():
    simulate(df, i, df_result, 'quick', 'null', next_counts, 0, lines, step_comp,line_next,step)
def action2():
    simulate(df, i, df_result, 'quick_comp', 'null', next_counts, 0, lines, step_comp,line_next,step)
def action3():
    simulate(df, i, df_result, 'quick_aug',pidselect,next_counts,0,lines,step_comp,line_next,step,half_chunk)
def action4():
    simulate(df, i, df_result, 'quick_aug_comp', pidselect,next_counts,0,lines,step_comp,line_next,step,half_chunk)
parser=argparse.ArgumentParser()
parser.add_argument('--number',required=True)
parser.add_argument('--s_list')
parser.add_argument('--t')
parser.add_argument('--seed',required=True)
parser.add_argument('--aug')
parser.add_argument('--pattern',required=True)
args=parser.parse_args()
column=['token','line','column','semantic','controlnum','feature','complexity','section','stimuli','pid']
slist=args.s_list.split(' ')
#print(slist)
tlist=[args.t]
f_list=[f for f in os.listdir('all_scanpath_repeat') if (f.split('.')[0][-3:]=='all' and f.split('_')[1] in [i[4:] for i in slist])] 
train_complexity_val=train_complexity(slist)
seed_value = int(args.seed)
np.random.seed(seed_value)
augnum = args.aug if (args.aug is not None and len(str(args.aug)) >= 1) else '91'
number=int(args.number)
random.seed(seed_value)
tname='x'.join([i[4:] for i in tlist])
df=pd.DataFrame(columns=column)
for f in f_list:
  df_f=pd.read_csv("all_scanpath_repeat/"+f)
  df=pd.concat([df,df_f])
  df.reset_index(drop=True, inplace=True)
if args.pattern=='aug':
  for t in tlist:
    target_stim="stim_info/"+t+'_sem.csv'
    with open("stim_info/"+t+'.cpp', 'r') as file:
      code = file.read()      
    lines = code.strip().split('\n')
    df_t=pd.read_csv(target_stim)
    df_e=pd.read_csv("all_scanpath_repeat/sti_"+str(t[4:])+"_all.csv")
    pidlist=list(set(df_e['pid'].to_list()))
    step_length=[]
    for sti in list(set(df['stimuli'].to_list())):
      df_sti=df[df['stimuli']==sti]
      for s in list(set(df_sti['pid'].to_list())):
        step_length.append(len(df_sti[df_sti['pid']==s]))    
    chunks_pair = select_chunks_half(pidlist, augnum,seed_value)
    for half_chunk, chunk in chunks_pair:
      print(half_chunk, chunk)
      pidselect = [j for j in pidlist if j in chunk]
      evaselect=[]
      df_stm=stm(pidselect)
      next_counts,short_next,line_next=generate_path(df,df_stm,5)
#      print(line_next)
      for pid in pidselect:
        evaselect.append(df_stm[df_stm['pid']==pid]['line'].to_list())
      for num in range(number):
        evalist=[]
        if len(evaselect)==0:
          continue
        step=set_step_aug(df,step_length,evaselect)
#        print(step,evaselect)
        step_comp=[]
 #       print(step)
#  print(step)
        columns=['node','Line','Column','semanic','controlnum','feature','complexity','corr_section','section','memtype','pidselect','half_pid']
        df_result=pd.DataFrame(columns=columns)
        if(3<step<10):
          for i in range(1,step+1):
            simulate(df_stm,i,df_result,'short_path',pidselect,next_counts,short_next,lines,step_comp,line_next,step,half_chunk)
        elif(step<=3):
          for i in range(1,step+1):
#             simulate(df,i,df_result,'quick_aug',pidselect,next_counts,0,lines,step_comp,line_next,step)  
            choice = random.choices([action3, action4], [0.5,0.5])[0]
            choice()
        else:
          for i in range(1,step+1):
            simulate(df_stm,i,df_result,'short_path',pidselect,next_counts,short_next,lines,step_comp,line_next,step,half_chunk)
        out=f'cross_valid_simu{augnum[0]}/'+t+"_"+tname+"_"+str(num)+"_aug"+str(augnum)+"_chunk"+str(augnum[1])+"_seed"+str(seed_value)+"_simu.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df_result.to_csv(out,index=False)
if args.pattern=='zero':
  for t in tlist:
    with open("stim_info/"+t+'.cpp', 'r') as file:
      code = file.read()      
    lines = code.strip().split('\n')
    target_stim="stim_info/"+t+'_sem.csv'
    df_t=pd.read_csv(target_stim)
    unique_df = df_t.drop_duplicates(subset='Line')
    test_complexity= unique_df['complexity'].sum()
    next_counts=generate_path(df,0,0)
    line_next={}
#    print(next_counts)
    for num in range(int(args.number)):
      step,steplist,comp_measure=set_step_zero(df,train_complexity_val,test_complexity)
      step_comp=random.choice(steplist)
      print(step)
#      print(step,steplist)
      columns=['node','Line','Column','semanic','controlnum','feature','complexity','corr_section','section','memtype','pidselect']
      df_result=pd.DataFrame(columns=columns)
      if(step>20):
        for i in range(1,step+1):
          simulate(df,i,df_result,'null','null',next_counts,0,lines,step_comp,line_next,step)
      elif(step<=3):
        for i in range(1,step+1):
          choice = random.choices([action1, action2], [1-comp_measure,comp_measure])[0]
          choice()
#          simulate(df,i,df_result,'quick','null',next_counts,0,lines,step_comp)      
      else:
        for i in range(1,step+1):
          simulate(df,i,df_result,'null','null',next_counts,0,lines,step_comp,line_next,step)   
      out='cross_valid_simu/'+t+"_"+tname+"_"+str(num)+"_aug"+'0'+"_chunk"+'0'+"_seed"+str(seed_value)+"_simu.csv"
      os.makedirs(os.path.dirname(out), exist_ok=True)
      df_result.to_csv(out,index=False)

    
    
  
