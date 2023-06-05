import networkx as nx
import matplotlib.pyplot as plt
import io
import numpy as np
import copy
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataset.ag_det2 import ag_categories, ag_rel_classes


def append_to_file(fpath, txt):
    with open(fpath, "a") as myfile:
        myfile.write(txt)

def print_dict(d, prefix=""):
    for k, v in d.items():
        kk = prefix + k
        if isinstance(v, dict):
            print_dict(v, kk+"/")
        elif isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
            print(kk, v)
        elif isinstance(v, list):
            print(kk, len(v))
        else:
            print(kk, v.shape)


def draw_graph2(objs, obj_mapping, rels, rel_mapping):
    plt.clf()

    g = nx.Graph()
    for o in objs:
        g.add_node(o)
    for (a, b) in (rels):
        g.add_edge(a, b)
    pos = nx.spring_layout(g)
    nx.draw(
        g, pos, edge_color='black', width=1, linewidths=1,
        node_size=1500, node_color='pink', alpha=0.9,
        labels=obj_mapping
    )
    nx.draw_networkx_edge_labels(
        g, pos,
        edge_labels=rel_mapping,
        font_color='red'
    )
    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='raw', dpi=100)
    fig = plt.gcf()
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr



def visualize_detection(sg_mode, batch, pred, idxs):

    ret = []
    for i in idxs:

        om = pred['box2im_idx'] == i
        # rm = pred['im_idx'] == i
        pred_scores = pred['pred_scores'][om].cpu()
        pred_labels = pred['pred_labels'][om].cpu()
        img = batch[i]['image']

        if sg_mode in ['predcls', 'sgcls']:
            instances = copy.deepcopy(batch[i]['instances'].to('cpu'))
            instances.pred_boxes = instances.gt_boxes
        else:
            instances = copy.deepcopy(batch['det_instances'][i].to('cpu'))

        instances.pred_classes = pred_labels
        instances.scores = pred_scores
        visualizer = Visualizer(img.permute(1,2,0).cpu().numpy()[:, :, ::-1], MetadataCatalog.get('ag_val'), scale=1.2)
        vis_out = visualizer.draw_instance_predictions(instances)
        det_img = vis_out.get_image()[:, :, ::-1]
        ret.append(det_img)
    return ret


def visualize_graph_predictions(pred, idxs, vis_gt=False):
    attention_relationships = ag_rel_classes[0:3]
    spatial_relationships = ag_rel_classes[3:9]
    contacting_relationships = ag_rel_classes[9:]
    global ag_categories

    ret = []
    for i in idxs:

        if not vis_gt:
            om = pred['box2im_idx'] == i
            rm = pred['im_idx'] == i
            pred_labels = pred['pred_labels'][om].cpu().numpy()
            candidate_rels = pred['pair_idx'][rm].cpu().numpy()
            
            atten_labels = pred['attention_distribution'][rm].cpu().argmax(-1).cpu().numpy()
            
            spati_rels = (pred['spatial_distribution'][rm] > 0.5).cpu().numpy() # r x NS
            spati_rels = [np.nonzero(sr)[0] for sr in spati_rels]

            conta_rels = (pred['contacting_distribution'][rm] > 0.5).cpu().numpy() # r x NC
            conta_rels = [np.nonzero(sr)[0] for sr in conta_rels]

        else:
            om = pred['gt_box2im_idx'] == i
            rm = pred['gt_im_idx'] == i
            pred_labels = pred['gt_labels'][om].cpu().numpy()
            candidate_rels = pred['gt_pair_idx'][rm].cpu().numpy()
            
            atten_labels = [sr[0] for bla, sr in zip(rm, pred['attention_gt']) if bla]
            spati_rels = [sr for bla, sr in zip(rm, pred['spatial_gt']) if bla]
            conta_rels = [sr for bla, sr in zip(rm, pred['contacting_gt']) if bla]


        objs = np.nonzero(om.cpu().numpy())[0]
        obj_mapping = {o: ag_categories[l] for o, l in zip(objs, pred_labels)}
        draw_rels = []
        rel_mapping = {}
        for j, rel in enumerate(candidate_rels):
            rellab = attention_relationships[atten_labels[j]]

            for k in spati_rels[j]:
                rellab += " \n " + spatial_relationships[k]
            
            for k in conta_rels[j]:
                rellab += " \n " + contacting_relationships[k]
            
            rel_mapping[tuple(rel)] = rellab
            draw_rels.append(tuple(rel))
        
        img = draw_graph2(objs, obj_mapping, draw_rels, rel_mapping)
        ret.append(img)
    return ret