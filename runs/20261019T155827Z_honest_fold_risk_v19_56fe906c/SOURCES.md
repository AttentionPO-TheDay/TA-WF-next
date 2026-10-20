# 来源

methods.py与maintenance_base.py显式复制自v18，仅本run局部导入；v19不调用其中fit函数。{
  "methods.py": "d16addbe3b43b520386cb72aec002687ac22e0b7cf3ed37a6a35d31d4ea15536",
  "maintenance_base.py": "bf054a3fb7007821a365217b7f46b61831ec03fcec574076ac1a7ff1936cda99"
}
原始特征/标签支持复用v17，折模型/OOF复用v18；所有依赖均通过config指定并核验已审计清单，不导入旧项目。
