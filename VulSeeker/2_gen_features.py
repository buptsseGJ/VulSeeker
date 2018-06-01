#!/usr/bin/python
# -*- coding: UTF-8 -*-
import config
import config_for_feature
import networkx as nx
import idaapi
import idautils
import idc
import sys
import os
import time
import shutil
from miasm2.core.bin_stream_ida import bin_stream_ida
from miasm2.core.asmblock import expr_is_label, AsmLabel, is_int
from miasm2.expression.simplifications import expr_simp
from miasm2.analysis.data_flow import dead_simp
from miasm2.ir.ir import AssignBlock, IRBlock
from utils import guess_machine, expr2colorstr
import re

idaapi.autoWait()

bin_num = 0
func_num = 0
function_list_file = ""
function_list_fp = None
functions=[]#由于windows文件名不区分大小写，这里记录已经分析的函数名（全部转换成小写，若重复，则添加当前时间戳作为后缀）

curBinNum = 0

class bbls:
	id=""
	define=[]
	use=[]
	defuse={}
	fathernode=set()
	childnode=set()
	define=set()
	use=set()
	visited=False

def calConstantNumber(ea):
	i = 0;
	curStrNum = 0
	numeric = 0
	#print idc.GetDisasm(ea)
	while i <= 1:
		if (idc.GetOpType(ea,i ) == 5):
			addr = idc.GetOperandValue(ea, i)
			if (idc.SegName(addr) == '.rodata') and (idc.GetType(addr) == 'char[]') and (i == 1):
				curStrNum = curStrNum + 1
			else :
				numeric = numeric + 1
		i = i + 1
	return numeric,curStrNum;

# 计算基本块的非结构性特征
def calBasicBlockFeature_vulseeker(block):
	StackNum = 0  # stackInstr
	MathNum = 0	 # arithmeticInstr
	LogicNum = 0  # logicInstr
	CompareNum = 0	# compareInstr
	ExCallNum = 0  # externalInstr
	InCallNum = 0  # internalInstr
	ConJumpNum = 0	# conditionJumpInstr
	UnConJumpNum = 0  # unconditionJumpInstr
	GeneicNum = 0  # genericInstr
	curEA = block.startEA
	while curEA <= block.endEA :
		inst = idc.GetMnem(curEA)
		if inst in config_for_feature.VulSeeker_stackInstr:
			StackNum = StackNum + 1
		elif inst in config_for_feature.VulSeeker_arithmeticInstr:
			MathNum = MathNum + 1
		elif inst in config_for_feature.VulSeeker_logicInstr:
			LogicNum = LogicNum + 1
		elif inst in config_for_feature.VulSeeker_compareInstr:
			CompareNum = CompareNum + 1
		elif inst in config_for_feature.VulSeeker_externalInstr:
			ExCallNum = ExCallNum + 1
		elif inst in config_for_feature.VulSeeker_internalInstr:
			InCallNum = InCallNum + 1
		elif inst in config_for_feature.VulSeeker_conditionJumpInstr:
			ConJumpNum = ConJumpNum + 1
		elif inst in config_for_feature.VulSeeker_unconditionJumpInstr:
			UnConJumpNum = UnConJumpNum + 1
		else:
			GeneicNum = GeneicNum + 1

		curEA = idc.NextHead(curEA,block.endEA)
		# elif inst in genericInstr:
		#	  GeneicNum = GeneicNum + 1
		# else:
		#	  print "+++++++++", inst.insn.mnemonic,
	fea_str =  str(StackNum) + "," + str(MathNum) + "," + str(LogicNum) + "," + str(CompareNum) + "," \
			  + str(ExCallNum) + "," + str(ConJumpNum) + "," + str(UnConJumpNum) + "," + str(GeneicNum) + ","
	return fea_str

#	字符常量的数量 , 数值常量的数量, 转移指令的数量,	 调用的数量,	  指令的数量,  算术指令的数量
def calBasicBlockFeature_gemini(block):
	numericNum = 0
	stringNum = 0
	transferNum = 0
	callNum = 0
	InstrNum = 0
	arithNum = 0
	logicNum = 0
	curEA = block.startEA
	while curEA <= block.endEA :
		#	数值常量 , 字符常量的数量
		numer, stri = calConstantNumber(curEA)
		numericNum = numericNum + numer
		stringNum = stringNum + stri
		#	转移指令的数量
		if idc.GetMnem(curEA) in config_for_feature.Gemini_allTransferInstr:
			transferNum = transferNum + 1
		# 调用的数量
		if idc.GetMnem(curEA) == 'call':
			callNum = callNum + 1
		# 指令的数量
		InstrNum = InstrNum + 1
		#	算术指令的数量
		if idc.GetMnem(curEA) in config_for_feature.Gemini_arithmeticInstr:
			arithNum = arithNum + 1
		#  逻辑指令
		if idc.GetMnem(curEA) in config_for_feature.Gemini_logicInstr:
			logicNum = logicNum + 1

		curEA = idc.NextHead(curEA,block.endEA)

	fea_str = str(numericNum) + ","+str(stringNum) + ","+str(transferNum) + ","+str(callNum) + ","+str(InstrNum) + ","+str(arithNum) + ","+str(logicNum) + ","
	return fea_str

def block_fea(allblock,fea_fp):
	for block in allblock:
		gemini_str = calBasicBlockFeature_gemini(block)
		vulseeker_str = calBasicBlockFeature_vulseeker(block)
		fea_str = str(hex(block.startEA)) + "," + gemini_str + vulseeker_str + "\n"
		fea_fp.write(fea_str)

def build_dfg(DG,IR_blocks):
	IR_blocks_dfg=IR_blocks
	#所有的被定义过的变量集合
	# alldefinedvar=set()
	# 起始节点
	startnode=''
	linenum=0
	for in_label, in_value in IR_blocks.items():
		linenum=0
		addr = in_label.split(":")[1].strip()
		# addr="0x"+in_label.split(":")[1].strip()[2:].lstrip('0')
		# 基本块结构体初始化
		tempbbls = bbls()
		tempbbls.id=addr
		tempbbls.childnode=set()
		tempbbls.fathernode=set()
		# 字典：记录等号左边的和等号右边的
		tempbbls.defuse={}
		# 字典：记录被定义的变量和被使用的变量 初始定义位置和最终定义位置
		tempbbls.defined={}
		tempbbls.used = {}
		# 集合：记录基本块中所有被定义过的变量
		tempbbls.definedset = set()
		tempbbls.visited=False
		IR_blocks_dfg[addr] = tempbbls

		for i in in_value:
			linenum+=1
			# 分析每一行代码
			# print i
			if '=' not in i or "call" in i or 'IRDst' in i:
				continue

			define = i.split('=')[0].strip()
			if '[' in define:
				define=define[define.find('[')+1:define.find(']')]
			use = i.split('=')[1].strip()
			if define not in tempbbls.defined:
				tempbbls.defined[define]=[linenum,0]
			else:
				tempbbls.defined[define][1]=linenum

			if define not in IR_blocks_dfg[addr].defuse:
				IR_blocks_dfg[addr].defuse[define] = set()

			# 如果没有括号，认为是单纯赋值
			if '(' not in use and '[' not in use:
				IR_blocks_dfg[addr].defuse[define].add(use)
				if use not in tempbbls.used:
					tempbbls.used[use] = [linenum, 0]
				else:
					tempbbls.used[use][1] = linenum
			#去括号
			else:
				srclist = list(i)
				for i in range(len(srclist)):
					if srclist[i] == ")" and srclist[i - 1] != ")":
						tmp = srclist[0:i + 1][::-1]
						for j in range(len(tmp)):
							if tmp[j] == "(":
								temps = "".join(srclist[i - j:i + 1])
								if temps.count(')') == 1 and temps.count('(') == 1:
									temps = temps[1:-1]	 # 不要括号
									IR_blocks_dfg[addr].defuse[define].add(temps)
									if temps not in tempbbls.used:
										tempbbls.used[temps] = [linenum, 0]
									else:
										tempbbls.used[temps][1] = linenum

								break

				for i in range(len(srclist)):
					if srclist[i] == "]" and srclist[i - 1] != "]":
						tmp = srclist[0:i + 1][::-1]
						for j in range(len(tmp)):
							if tmp[j] == "[":
								temps = "".join(srclist[i - j:i + 1])
								if temps.count(']') == 1 and temps.count(']') == 1:
									temps = temps[1:-1]	 # 不要括号
									IR_blocks_dfg[addr].defuse[define].add(temps)
									if temps not in tempbbls.used:
										tempbbls.used[temps] = [linenum, 0]
									else:
										tempbbls.used[temps][1] = linenum
								break

		# print "addr",addr
		# print "IR_blocks_dfg",IR_blocks_dfg
		# print "IR_blocks_dfg[addr].defuse",IR_blocks_dfg[addr].defuse

	for cfgedge in DG.edges():
		innode=str(cfgedge[0])
		outnode=str(cfgedge[1])
		# print "in out**"+innode+"**"+outnode
		if innode==outnode:
			continue
		if IR_blocks_dfg.has_key(innode):
			IR_blocks_dfg[innode].childnode.add(outnode)
		if IR_blocks_dfg.has_key(outnode):
			IR_blocks_dfg[outnode].fathernode.add(innode)

	# 找起始节点，记录每个基本块中定义的所有变量
	cfg_nodes = DG.nodes()
	# print "CFG nodes find father ",len(cfg_nodes)
	# startnode = list(IR_blocks_dfg.keys())[0]
	startnode = None
	for addr,bbloks in IR_blocks_dfg.items():
		if ':' in addr:
			continue
		if len(cfg_nodes)==1 or startnode is None :#只有一个基本块	 或	形成整环
			startnode = addr
		# print addr,addr in cfg_nodes,IR_blocks_dfg[addr].fathernode
		if addr in cfg_nodes and len(IR_blocks_dfg[addr].fathernode)==0:
			startnode=addr
		for definevar in IR_blocks_dfg[addr].defuse:
			IR_blocks_dfg[addr].definedset.add(definevar)
	# print "startnode	:",startnode
	if startnode is None:
		return nx.DiGraph()
	else:
		return gen_dfg(IR_blocks_dfg,startnode)

def gen_dfg(IR_blocks_dfg,startnode):
	# DFS遍历
	res_graph = nx.DiGraph()
	# cur_step = 0
	stack_list = []
	visited ={}
	# v2代表要访问第二次但是还没访问的
	visited2= {}
	# v3表示已经访问两次结束的
	visited3 = {}
	for key,val in IR_blocks_dfg.items():
		visited2[key]=set()
		visited3[key]=set()
	visitorder=[]
	# print "Visit!!", startnode
	# print "startnode!!",startnode
	IR_blocks_dfg[startnode].visited=True
	visited[startnode] = '1'
	visitorder.append(startnode)
	stack_list.append(startnode)
	while len(stack_list) > 0:
		cur_node = stack_list[-1]
		next_nodes = set()
		if IR_blocks_dfg.has_key(cur_node):
			next_nodes = IR_blocks_dfg[cur_node].childnode
		# print len(stack_list),cur_node,"-->",next_nodes
		if len(next_nodes) == 0:  # 叶子节点要回退
			stack_list.pop()
			visitorder.pop()
			# cur_step = cur_step - 1
		else:
			if (len(set(next_nodes) - set(visited.keys())) == 0 ) and len(next_nodes & visited2[cur_node])==0:
				# 如果都被访问过 要回退
				stack_list.pop()
				visitorder.pop()

			else:
				for i in next_nodes:
					if i not in visited or i in visited2[cur_node]:
						fathernodes=set()
						usevar = {}
						defined={}
						if IR_blocks_dfg.has_key(i):
							# 列表：父节点
							fathernodes=IR_blocks_dfg[i].fathernode
							# 字典：基本块中使用的变量 出现的位置
							usevar=IR_blocks_dfg[i].used
							# 字典：基本块中定义的变量 出现的位置
							definevar=IR_blocks_dfg[i].defined
						# 集合：第一层父亲节点中定义的变量
						fdefinevarset=set()
						# 布尔值：记录是否在父节点找到
						findflag=False
						# 集合：所有父亲节点们定义的变量
						allfdefinevarset=set()

						for uvar in usevar:
							# 如果这个使用的变量没有在自基本块中被定义或使用的位置在定义的位置之前
							# 去父节点找
							if uvar not in definevar or usevar[uvar][0] < definevar[uvar][0]:
								for fnode in fathernodes:
									fdefinevarset = set()
									if IR_blocks_dfg.has_key(fnode):
										fdefinevarset = IR_blocks_dfg[fnode].definedset
									allfdefinevarset|=fdefinevarset
									if uvar in fdefinevarset:
										res_graph.add_edge(fnode, i)
										print fnode,'->',i,"var:",uvar
								# 可能存在和父亲的父亲节点之间的数据依赖,按照深度优先的遍历顺序反向找
								for j in range(len(visitorder)-1,-1,-1):
									visitednode=visitorder[j]
									temp_definedset = set()
									if IR_blocks_dfg.has_key(visitednode):
										temp_definedset = IR_blocks_dfg[visitednode].definedset
									if uvar in temp_definedset - allfdefinevarset:
										res_graph.add_edge(visitednode, i)
										allfdefinevarset|=temp_definedset
										print "fffff", visitednode, '->', i, "var:", uvar

						visited[i] = '1'
						visitorder.append(i)
						if i in visited2[cur_node]:
							visited2[cur_node].remove(i)
							visited3[cur_node].add(i)
						temp_childnode = set()
						if IR_blocks_dfg.has_key(i):
							temp_childnode = IR_blocks_dfg[i].childnode
						visited2[cur_node] |=(set(temp_childnode) & set(visited) )-set(visited3[cur_node])
						stack_list.append(i)
	return res_graph

def get_father_block(blocks, cur_block, yes_keys):
	# print "find father block",cur_block.label
	father_block = None
	for temp_block in blocks:
		# print temp_block.get_next(),"<>",cur_block.label
		if temp_block.get_next() is cur_block.label:
			father_block = temp_block
	if father_block is None:
		return None
	is_Exist = False
	for yes_label in yes_keys:
		# print father_block
		# print father_block.label
		if ((str(father_block.label) + "L")).split(' ')[0].endswith(yes_label):
			is_Exist = True
	if not is_Exist:
		# print "Not exist", ((str(father_block.label) + "L")).split(' ')[0]
		father_block = get_father_block(blocks, father_block, yes_keys)
		return father_block
	else:
		# print "exist", ((str(father_block.label) + "L")).split(' ')[0]
		return father_block

def rebuild_graph(cur_block, blocks, IR_blocks, no_ir):
	# print ">>rebuild ", len(no_ir)
	yes_keys = list(IR_blocks.keys())
	no_keys = list(no_ir.keys())
	next_lable = (str(cur_block.label) + "L").split(' ')[0]
	father_block = get_father_block(blocks, cur_block, yes_keys)
	if not father_block is None:
		for yes_label in yes_keys:
			if ((str(father_block.label) + "L")).split(' ')[0].endswith(yes_label):
				for no_label in no_keys:
					# print "222", next_lable, no_label
					if next_lable.endswith(no_label):
						IR_blocks[yes_label].pop()
						IR_blocks[yes_label].extend(IR_blocks[no_label])
						# print "<<<del", no_label
						# print "<<<len", len(no_ir)
						del (no_ir[no_label])
						del (IR_blocks[no_label])
	return IR_blocks, no_ir

def dataflow_analysis(addr,block_items,DG):

	machine = guess_machine()
	mn, dis_engine, ira = machine.mn, machine.dis_engine, machine.ira
	#
	# print "Arch", dis_engine
	#
	# fname = idc.GetInputFile()
	# print "file name : ",fname
	# print "machine",machine

	bs = bin_stream_ida()
	mdis = dis_engine(bs)
	mdis.dont_dis_retcall_funcs=[]
	mdis.dont_dis=[]
	ir_arch = ira(mdis.symbol_pool)
	blocks = mdis.dis_multiblock(addr)
	for block in blocks:
		ir_arch.add_block(block)
		# print ">>asm block",block

	IRs = {}
	for lbl, irblock in ir_arch.blocks.items():
		insr = []
		for assignblk in irblock:
			for dst, src in assignblk.iteritems():
				insr.append(str(dst) + "=" + str(src))
		# print ">>ir",(str(lbl)+"L"),insr
		IRs[str(lbl).split(' ')[0]+"L"] = insr
	# print	 "IRs.keys()",IRs.keys()

	IR_blocks={}
	no_ir = {}
	# print "block_items",block_items
	for block in blocks:
		# print "block.label",block.label
		isFind = False
		item = str(block.label).split(' ')[0]+ "L"
		# item = "0x"+(str(block.label).split(' ')[0].split(':')[1][2:]).lstrip('0') + "L"
		# print "block line number",item
		for block_item in block_items:
			# print block_item
			if item.endswith(block_item):
				isFind = True

		# for irlabel in IRs.keys():
		#	  print irlabel,item
		#	  if irlabel.endswith(item):
		#		  itrm = irlabel
		# print item,IRs[str(block.label) + "L"]
		if IRs.has_key(item):
			if isFind:
				IR_blocks[item] = IRs[item]
			else:
				IR_blocks[item] = IRs[item]
				no_ir[item]= IRs[item]
	# print "yes_ir : ",list(IR_blocks.keys())
	no_keys = list(no_ir.keys())
	# print "no_ir : ",no_keys
	for cur_label in no_keys:
		cur_block = None
		# print ""
		# print ""
		# print "find no_ir	 label is : ",cur_label
		for block in blocks:
			#去除loc_0000000000413D4C:0x00413d4cL callXXX的情况
			temp_index = str(block.label).split(' ')[0]+"L"
			# print block.label,temp_index
			if temp_index.endswith(cur_label):
				cur_block = block
		if not cur_block is None:
			# print "find no_ir ",cur_block
			IR_blocks, no_ir = rebuild_graph(cur_block,blocks,IR_blocks,no_ir)
	# print len(no_ir)
	#
	# save_file = "C:/AppData/Setup/IDA_Pro_v6.8/" + idc.GetFunctionName(addr) + ".txt"
	# fp = open(save_file, 'w')
	# print "**********result*************"
	#
	# fp.write( str(DG.edges())	 + "\n")
	# for in_label, in_value in IR_blocks.items():
	#	  fp.write(in_label+"\n")
	#	  for i in in_value:
	#		  fp.write("\t\t"+i+"\n")
	IR_blocks_toDFG = {}
	for key, value in IR_blocks.items():
		if len(key.split(':'))>1:
			key = key.split(':')[0] + ":0x"+key.split(':')[1].strip()[2:].lstrip('0')
		# print "dg to dfg : ",key
		IR_blocks_toDFG[key] = value
	# print "IR_blocks_toDFG",IR_blocks_toDFG
	# print "CFG edges <<",DG.number_of_edges(),">> :",DG.edges()
	dfg = build_dfg(DG,IR_blocks_toDFG)
	dfg.add_nodes_from(DG.nodes())
	print "CFG edges <<",DG.number_of_edges(),">> :",DG.edges()
	print "DFG edges <<",dfg.number_of_edges(),">> :",dfg.edges()
	print "DFG nodes : ",dfg.number_of_nodes()
	return dfg

def main():
	global bin_num, func_num, function_list_file, function_list_fp,functions

	fea_path=""
	if len(idc.ARGV)<1:
		fea_path=config.FEA_DIR+"\\CVE-2015-1791\\DAP-1562_FIRMWARE_1.10"
		bin_path = config.O_DIR + "\\CVE-2015-1791\\DAP-1562_FIRMWARE_1.10\\wpa_supplicant.i64"
		binary_file = bin_path.split(os.sep)[-1]
		program = "CVE-2015-1791"
		version = "DAP-1562_FIRMWARE_1.10"
	else:
		print idc.ARGV[1]
		print idc.ARGV[2]
		fea_path_origion = idc.ARGV[1]
		fea_path_temp = idc.ARGV[1]+"\\temp"
		bin_path = idc.ARGV[2]
		binary_file = bin_path.split(os.sep)[-1]
		program = idc.ARGV[3]
		version = idc.ARGV[4]


	print "Directory path	：	", fea_path_origion
	function_list_file = fea_path_origion + os.sep + "functions_list_fea.csv"
	function_list_fp = open(function_list_file, 'w')  # a 追加

	textStartEA = 0
	textEndEA = 0
	for seg in idautils.Segments():
		if (idc.SegName(seg)==".text"):
			textStartEA = idc.SegStart(seg)
			textEndEA = idc.SegEnd(seg)
			break

	# 遍历文件中的所有指令，保存到文件
	# 生成dict，将指令地址与指令id一一对应
	print "遍历所有指令，生成instDict, inst_info"
	for func in idautils.Functions(textStartEA, textEndEA):
		# Ignore Library Code
		flags = idc.GetFunctionFlags(func)
		if flags & idc.FUNC_LIB:
			print hex(func), "FUNC_LIB", idc.GetFunctionName(func)
			continue

		cur_function_name = idc.GetFunctionName(func)
		print cur_function_name
		#if cur_function_name != "X509_NAME_get_text_by_NID":
		#	 continue
		
		fea_path = fea_path_origion
		if cur_function_name.lower() in functions:
			fea_path = fea_path_temp
			if not os.path.exists(fea_path):
				os.mkdir(fea_path)
			#cur_function_name = cur_function_name + "_"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
		functions.append(cur_function_name.lower())
		print cur_function_name, "=====start"# 打印函数名
		'''	  
			记录函数的控制流信息,生成CFG邻接表
			每个txt中存放一个函数的控制流图, 命名方式:[函数名_cfg.txt]
			# a b c	 # a-b a-c
			# d e  # d-e
			# G = nx.read_adjlist(‘test.adjlist’)
		'''
		allblock = idaapi.FlowChart(idaapi.get_func(func))
		cfg_file = fea_path + os.sep + str(cur_function_name)+ "_cfg.txt"
		cfg_fp = open(cfg_file, 'w')
		block_items = []
		DG = nx.DiGraph()
		for idaBlock in allblock:
			temp_str = str(hex(idaBlock.startEA))
			block_items.append(temp_str[2:])
			DG.add_node(hex(idaBlock.startEA))
			for succ_block in idaBlock.succs():
				DG.add_edge(hex(idaBlock.startEA),hex(succ_block.startEA))
			for pred_block in idaBlock.preds():
				DG.add_edge(hex(pred_block.startEA),hex(idaBlock.startEA))
		# print DG.edges()
		# print block_items
		for cfg_node in DG.nodes():
			# print cfg_node
			cfg_str = str(cfg_node)
			for edge in DG.succ[cfg_node]:
				cfg_str = cfg_str + " " + edge
					# print hex(edge.addr),
					# print hex(cfg_node.addr, create_using=nx.DiGraph()),"---->",hex(edge.addr)  # 遍历所有边
			# print "cfg_str",cfg_str
			cfg_str = cfg_str + "\n"
			cfg_fp.write(cfg_str)

		'''
			记录函数的数据流信息生成DFG邻接表
			每个txt中存放一个函数的数据流图, 命名方式:[函数名_dfg.txt]
			# a b c	 # a-b a-c
			# d e  # d-e
			# G = nx.read_adjlist(‘test.adjlist’)
		'''
		dfg = dataflow_analysis(func,block_items,DG)
		dfg_file = fea_path + os.sep + str(cur_function_name)+ "_dfg.txt"
		dfg_fp = open(dfg_file, 'w')
		for dfg_node in dfg.nodes():
			dfg_str = dfg_node
			for edge in dfg.succ[dfg_node]:
				dfg_str = dfg_str + " " + edge
			# print "dfg_str: ",dfg_str
			dfg_str = dfg_str + "\n"
			dfg_fp.write(dfg_str)

		'''
			记录函数的基本块信息,抽取一个函数中各个基本块的特征
			每个函数保存成一个CSV文件, 命名方式:[函数名_fea.csv]
			#	堆栈、算术、逻辑、比较、外部调用、内部调用、条件跳转、非条件跳转、普通指令
		'''
		fea_file = fea_path + os.sep + str(cur_function_name)+ "_fea.csv"
		fea_fp = open(fea_file, 'w')
		block_fea(allblock,fea_fp)

		'''
			记录所有函数的原始反汇编指令
		orig_file = fea_path + os.sep + str(cur_function_name)+ "_origion_instruction_info.csv"
		orig_fp = open(orig_file, 'w')
		inst_file = fea_path + os.sep + str(cur_function_name)+"_instruction_info.csv"
		inst_fp = open(inst_file, 'w')
		for instru in idautils.FuncItems(func):
			orig_fp.write(hex(instru)+","+idc.GetDisasm(instru)+"\n")
			inst_fp.write(idc.GetMnem(instru)+"\n")
		'''

		'''	  
			记录函数概要信息，函数名，路径，基本块数量，控制流边的数量，数据流边的数量
		'''

		print cur_function_name, "=====finish"	# 打印函数名
		# 函数名,基本块数量,数据流结点数量,控制流边数量,数据流边数量,所在程序名,版本编号,所在二进制文件路径
		function_str = str(cur_function_name) + "," + str(DG.number_of_nodes()) +  "," + \
						   str(DG.number_of_edges()) + "," + str(dfg.number_of_edges()) +  "," + \
						   str(program) + "," + str(version) + "," +str(bin_path) + ",\n"
		function_list_fp.write(function_str)
	return

#redirect output into a file, original output is the console.
def stdout_to_file(output_file_name, output_dir=None):
	if not output_dir:
		output_dir = os.path.dirname(os.path.realpath(__file__))
	output_file_path = os.path.join(output_dir, output_file_name)
	print output_file_path
	print "original output start"
	# save original stdout descriptor
	orig_stdout = sys.stdout
	# create output file
	f = file(output_file_path, "w")
	# set stdout to output file descriptor
	sys.stdout = f
	return f, orig_stdout


if __name__=='__main__':
	f, orig_stdout = stdout_to_file("output_"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+".txt")
	main()
	print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	sys.stdout = orig_stdout #recover the output to the console window
	f.close()

	idc.Exit(0)
