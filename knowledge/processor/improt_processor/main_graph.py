"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""
from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.improt_processor.state import ImportGraphState
from knowledge.processor.improt_processor.nodes.entry_node import EntryNode
from knowledge.processor.improt_processor.nodes.md_to_img_node import MarkdownToImageNode
from knowledge.processor.improt_processor.nodes.pdf_to_md_node import PdfToMdNode


def import_route(state: ImportGraphState) -> str:
    """
    根据state中的 is_pdf_read_enabled is_md_read_enabled 决定如何到达下一个节点
    Returns:

    """
    # 1. 获取上传的文件属于pdf or md
    if state.get('is_pdf_read_enabled'):
        return "pdf_to_md_node"
    if state.get('is_md_read_enabled'):
        return "md_to_img_node"
    return END




def improt_graph():
    """
    定义编排节点
    :return:
    """

    work_flow = StateGraph(state_schema=ImportGraphState)

    #添加节点
    work_flow.set_entry_point("entry_node")

    #定义节点名和节点实例映射

    node_name_obj = {
        "entry_node": EntryNode(),
        "md_to_img_node": MarkdownToImageNode(),
        "pdf_to_md_node": PdfToMdNode(),

    }
    #添加节点
    for node_name,node_obj in node_name_obj.items():
        work_flow.add_node(node_name,node_obj)

    #添加条件边
    work_flow.add_conditional_edges(
        "entry_node",
        import_route,
        {
            "pdf_to_md_node":"pdf_to_md_node",
            "md_to_img_node":"md_to_img_node",
            END: END
        })
    # 5.2 定义业务边
    work_flow.add_edge("pdf_to_md_node", "md_to_img_node")
    work_flow.add_edge("md_to_img_node", END)

    # 5.3 编译
    complied_state_graph = work_flow.compile()

    return complied_state_graph




