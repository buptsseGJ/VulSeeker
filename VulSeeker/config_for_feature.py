#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 转移指令
Gemini_allTransferInstr = ['je', 'jz', 'jne', 'jnz', 'js', 'jns', 'jo', 'jno', 'jc', 'jnc', 'jp', 'jpe', 'jnp', 'jpo',
                           'jl', 'jpo', 'jl', 'jnge', 'jnl', 'jge', 'jg', 'jnle', 'jng', 'jle', 'jb', 'jnae', 'jnb',
                           'jae', 'ja', 'jnbe', 'jna', 'jbe', 'jmp', 'opd', 'jmp opd', 'jcxz', 'jecxz', 'loop', 'loopw',
                           'loopd', 'loope', 'loopz', 'loopne', 'loopnz', 'reg', 'ops', 'bound', 'bound reg', 'call',
                           'opd', 'call opd', 'ret', 'retn', 'int', 'into', 'iret', 'iretd', 'iretf', 'j', 'jal',
                           'jalr', 'b', 'bal', 'bl', 'blx', 'bx', 'bc0f', 'bc0f1', 'bc0t', 'bc0t1', 'bc2f', 'bc2f1',
                           'bc2t', 'bc2t1', 'bc1f', 'bc1f1', 'bc1t', 'bc1t1', 'beq', 'beq1', 'beqz', 'beqz1', 'bge',
                           'bge1', 'bgeu', 'bgeu1', 'bgez', 'bgez1', 'bgt', 'bgt1', 'bgtu', 'bgtu1', 'bgtz', 'bgtz1',
                           'ble', 'ble1', 'bleu', 'bleu1', 'blez', 'blez1', 'blt', 'blt1', 'bltu', 'bltu1', 'bltz',
                           'bltz1', 'bne', 'bnel', 'bnez', 'bnezl', 'bgeza1', 'bgeza11', 'bltza1', 'bltza11']
# 算数指令
Gemini_arithmeticInstr = ['aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addu', 'addiu', 'dadd', 'daddi', 'daddu', 'daddiu',
                          'dsub', 'dsubu', 'subu', 'abs', 'dabs', 'dneg', 'dnegu', 'negu', 'cbw', 'cdq', 'cwd', 'cwde',
                          'daa', 'das', 'dec', 'div', 'divo', 'divou', 'idiv', 'ddiv', 'ddivu', 'divu', 'dmul', 'dmulu',
                          'mulo', 'mulou', 'dmulo', 'dmulou', 'dmult', 'dmultu', 'mult', 'multu', 'imul', 'inc', 'mul',
                          'drem', 'dremu', 'rem', 'remu', 'mfhi', 'mflo', 'mthi', 'mtlo', 'sbb', 'sub', 'rsb', 'sbc',
                          'rsc', 'c', 'r', 'mla', 'smull', 'smlal', 'umull', 'umlal']
Gemini_logicInstr = ['and', 'andi', 'or', 'xor', 'not', 'test', 'eor', 'orr', 'teq', 'tst', 'ori', 'nor']
# 转移指令
VulSeeker_allTransferInstr = ['reg', 'ops', 'bound', 'bound reg', 'int', 'into', 'iret', 'iretd', 'iretf', ]
# 堆栈指令
VulSeeker_stackInstr = ['push', 'pop', 'pusha', 'popa', 'pushad', 'popad', 'pushf', 'popf', 'popal', 'pushd', 'popd',
                        'stmfa', 'ldmfa', 'stmed', 'ldmed', 'stmea', 'ldmea', 'stm', 'ldm', 'ldp', 'stp', 'stmfd',
                        'ldmfd', 'fucompp', 'fucomp', 'fucompi', 'fucomi']
# 算数指令
VulSeeker_arithmeticInstr = ['xadd', 'aaa', 'aad', 'aam', 'aas', 'adc', 'add', 'addu', 'addiu', 'daa', 'dadd', 'adds',
                             'madd', 'addi', 'addiu', 'daddi', 'daddu', 'daddiu', 'dsub', 'dsubu', 'subu', 'abs',
                             'dabs', 'dneg', 'cneg', 'fadd', 'fsub', 'sub', 'subu', 'dnegu', 'negu', 'cbw', 'cdq',
                             'cwd', 'cwde', 'daa', 'das', 'dec', 'div', 'divo', 'udiv', 'fdiv', 'divu', 'fdivp',
                             'divou', 'idiv', 'ddiv', 'ddivu', 'divu', 'dmul', 'dmulu', 'neg', 'sdiv', 'smulh', 'msub',
                             'mul', 'mulu', 'mulo', 'mulou', 'dmulo', 'dmulou', 'dmult', 'dmultu', 'mult', 'multu',
                             'imul', 'fmul', 'inc', 'fmulp', 'nec', 'drem', 'dremu', 'rem', 'remu', 'mfhi', 'mflo',
                             'mthi', 'mtlo', 'subs', 'fdivp', 'sbb', 'rsb', 'rsblt', 'sbc', 'sbcs', 'sbcssbc', 'rsc',
                             'c', 'r', 'mla', 'smull', 'smlal', 'umull', 'umulh', 'umlal']
# 逻辑指令
VulSeeker_logicInstr = ['and', 'ands', 'andi', 'andeq', 'or', 'xor', 'not', 'eor', 'orr', 'teq', 'ori', 'nor', 'shl',
                        'sal', 'shr', 'orn', 'rev', 'revh', 'revsh', 'addss', 'subss', 'divss', 'mulss', 'divsd',
                        'mulsd', 'addsd', 'subsd', 'sar', 'rol', 'ror', 'rcl', 'rcr', 'lsl', 'lsr', 'asr', 'rrx', 'bic',
                        'movi', 'sxtb', 'sxth', 'sxtw', 'uxtb', 'uxth', 'xori', 'pxor', 'sll', 'sllv', 'srl', 'dsll',
                        'dsll32', 'dsrl', 'dsrl32', 'dsra', 'dsra32', 'dsllv', 'dsrlv', 'dsrav', 'sra', 'srav', 'srl',
                        'srlv']
# 逻辑指令
VulSeeker_segInstr = ['bfc', 'bfi', 'bfxil', 'sbfiz', 'sbfx', 'ubfiz', 'bfm', 'sbfm', 'ubfm']
# 比较指令
VulSeeker_compareInstr = ['test', 'tst', 'cmpxchg', 'cmp', 'cmn', 'fcmp', 'fcmpz', 'fcmpez', 'fcmpe', 'ccmp', 'cmpeq',
                          'ucomisd', 'comisd', 'csel', 'cset', 'csetm', 'cinc', 'cinv', 'csinc', 'csinv', 'slt', 'slti',
                          'sltu', 'sltui', 'sltiu', 'cmpsb', 'ucomiss']
# 调用
VulSeeker_externalInstr = ['blx', 'bx', 'call', 'callq', 'bl', 'bal']
VulSeeker_internalInstr = []
# 条件跳转
VulSeeker_conditionJumpInstr = ['jle', 'loop', 'loopw', 'loopd', 'loope', 'loope', 'loopz', 'loopne', 'loopnz', 'jcxz',
                                'jecxz', 'tbz', 'tbnz', 'cbz', 'cbnz', 'bls', 'blo', 'bhi', 'bhs', 'bcc', 'bmi', 'bvs',
                                'bpl', 'b.hi', 'b.eq', 'b.le', 'b.ne', 'b.cs', 'b.hs', 'b.cc', 'b.lo', 'b.mi', 'b.pl',
                                'b.vs', 'b.vc', 'b.ls', 'b.ge', 'b.lt', 'b.gt', 'b.al', 'beqz', 'beq', 'bnez', 'jalr',
                                'jal', 'bne', 'bnel', 'bnez', 'bnezl', 'beq1', 'beqz1', 'blez', 'blezl', 'blt', 'blt1',
                                'bltu', 'bltu1', 'bltz', 'bltza1', 'bltza11', 'ble', 'ble1', 'bleu', 'bleu1', 'bc0f',
                                'bc0f1', 'bc0t', 'bc0t1', 'bc2f', 'bc2f1', 'bc2t', 'bc2t1', 'bc1f', 'bc1f1', 'bc1t',
                                'bc1t1', 'bgeza11', 'bge', 'bge1', 'bgeu', 'bgeu1', 'bgez', 'bgez1', 'bgt', 'bgt1',
                                'bgtu', 'bgtu1', 'bgtz', 'bgtz1', 'bgeza1', 'je', 'jz', 'jne', 'jnz', 'js', 'jns', 'jo',
                                'jno', 'jc', 'jnc', 'jp', 'jpe', 'jnp', 'jpo', 'jl', 'jpo', 'jl', 'jnge', 'jnl', 'jge',
                                'jg', 'jnle', 'jng', 'jb', 'jnae', 'jnb', 'jae', 'ja', 'jnbe', 'jna', 'jbe', 'jcxz',
                                'jecxz']
# 非条件跳转
VulSeeker_unconditionJumpInstr = ['jmp', 'ret', 'retn', 'b', 'br', 'blr', 'j', 'jr', 'eret', 'bal', 'leave']
# 基本指令
VulSeeker_genericInstr = ['mov', 'move', 'movz', 'movn', 'movne', 'movk', 'fmov', 'mvn', 'movsx', 'movzx', 'movsd',
                          'movq', 'cmovle', 'cmovge', 'cmovl', 'movaps', 'movabs', 'cmovb', 'cmove', 'cmovne', 'movge',
                          'movlt', 'moveq', 'movlo', 'movhs', 'movgt', 'movle', 'movhi', 'movls', 'movsw', 'cdqe',
                          'movapd', 'movsd', 'cmovbe', 'cmovns', 'movsxd', 'movss', 'cmovae', 'ucvtf', 'fcvtzu',
                          'scvtf', 'cdq', 'bswap', 'xchg', 'xlat', 'trap', 'in', 'out', 'lea', 'lfs', 'lds', 'lgs',
                          'fld', 'fldz', 'fld1', 'fild', 'lw', 'lh', 'lb', 'ld', 'lwu', 'lhu', 'lbu', 'sb', 'sh', 'sd',
                          'sw', 'swr', 'swl', 'lui', 'fist', 'lwr', 'lwl', 'sdr', 'sdl', 'ldl', 'ldr', 'ldrls', 'ldrd',
                          'shrd', 'shls', 'shld', 'adrh', 'adrp', 'lss', 'lahf', 'sahf', 'ldrb', 'ldrp', 'ldrh',
                          'ldrsh', 'ldrsw', 'ldrsb', 'nop', 'adr', 'adrl', 'str', 'strb', 'strd', 'strp', 'strh', 'swp',
                          'swpb', 'ldur', 'ldurb', 'ldursb', 'ldurh', 'ldursh', 'ldursw', 'ldtr', 'ldtrb', 'ldtrsb',
                          'ldtrh', 'ldtrsh', 'ldtrsw', 'stur', 'sturb', 'sturh', 'fstp', 'fucom', 'fnstsw', 'fxch',
                          'sete', 'setne', 'seta', 'setb', 'setbe', 'setg', 'setl', 'setae', 'fnstcw', 'fldcw', 'fistp',
                          'rep stosd', 'rep stosq', 'rep movsd', 'repe cmpsb', 'repne scasb', 'rep movsq', 'bfc', 'bfi',
                          'bfxil', 'sbfiz', 'sbfx', 'ubfiz', 'bfm', 'sbfm', 'ubfm', 'ubfx', 'cvttss2si', 'cvtsi2ss',
                          'cvtsi2sd', 'cvttsd2si', 'cqo']
