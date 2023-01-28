import math

from data_type.constants import ALL_GATE
from data_type.enums import LayerType
from util.common import isNodeActive, isIntrinsicallyTrainableLayer


def updateNodeConcernForRnn(layer):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum):
            layer.inactive_count[nodeNum] += 1
        else:
            layer.active_count[nodeNum] += 1


def updateNodeConcernForRefLayer(updateLayer, referenceLayer, current_timestep=0, unrolled=False):
    for nodeNum in range(referenceLayer.num_node):
        if not unrolled:
            if not isNodeActive(referenceLayer, nodeNum, timestep=current_timestep):
                updateLayer.inactive_count[nodeNum] += 1
            else:
                updateLayer.active_count[nodeNum] += 1
        else:
            if not isNodeActive(referenceLayer, nodeNum, timestep=current_timestep):
                updateLayer.inactive_count[current_timestep][nodeNum] += 1
            else:
                updateLayer.active_count[current_timestep][nodeNum] += 1


def updateNodeConcernForRnnAtTimestep(layer, current_timestep):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum, timestep=current_timestep):
            if layer.unrolled:
                layer.inactive_count[current_timestep][nodeNum] += 1
            else:
                layer.inactive_count[nodeNum] += 1
        else:
            if layer.unrolled:
                layer.active_count[current_timestep][nodeNum] += 1
            else:
                layer.active_count[nodeNum] += 1


def updateNodeConcernForRegular(layer):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum):
            layer.inactive_count[nodeNum] += 1
    else:
        layer.active_count[nodeNum] += 1


def updateNodeConcernForOutputLayer(layer):
    for nodeNum in range(layer.num_node):
        layer.DW[:, nodeNum] = layer.W[:, nodeNum]
        layer.DB[nodeNum] = layer.B[nodeNum]


def updateNodeConcernForOutputLayerForRefLayer(updateLayer, referenceLayer, timestep, unrolled=False):
    for nodeNum in range(updateLayer.num_node):
        if not unrolled:
            updateLayer.DW[:, nodeNum] = referenceLayer.W[:, nodeNum]
            updateLayer.DB[nodeNum] = referenceLayer.B[nodeNum]
        else:
            updateLayer.DW[timestep][:, nodeNum] = referenceLayer.W[:, nodeNum]
            updateLayer.DB[timestep][nodeNum] = referenceLayer.B[nodeNum]


def isNodeRemoved(layer, current_timestep=None, nodeNum=None):
    if layer.unrolled and current_timestep is not None:
        if layer.DB[current_timestep][nodeNum] != 0:
            return False
        for x in layer.DW[current_timestep][:, nodeNum]:
            if x != 0:
                return False

    else:

        if layer.DB[nodeNum] != 0:
            return False
        for x in layer.DW[:, nodeNum]:
            if x != 0:
                return False
    return True


def removeNode(layer, current_timestep=None, nodeNum=None):
    if layer.unrolled and current_timestep is not None:
        layer.DW[current_timestep][:, nodeNum] = [0 for x in layer.DW[current_timestep][:, nodeNum]]
        if layer.DU is not None:
            layer.DU[current_timestep][:, nodeNum] = [0 for x in layer.DU[current_timestep][:, nodeNum]]
            if nodeNum < layer.DU[current_timestep].shape[0]:
                layer.DU[current_timestep][nodeNum, :] = [0 for x in layer.DU[current_timestep][nodeNum, :]]

        layer.DB[current_timestep][nodeNum] = 0

        if layer.next_layer is not None and isIntrinsicallyTrainableLayer(layer.next_layer):
            if layer.next_layer.unrolled:
                if nodeNum < layer.next_layer.DW[current_timestep].shape[0]:
                    layer.next_layer.DW[current_timestep][nodeNum, :] = [0 for x in
                                                                         layer.next_layer.DW[current_timestep]
                                                                         [nodeNum, :]]
            elif current_timestep == layer.timestep - 1:
                if nodeNum < layer.next_layer.DW.shape[0]:
                    layer.next_layer.DW[nodeNum, :] = [0 for x in layer.next_layer.DW[nodeNum, :]]

    else:
        layer.DW[:, nodeNum] = [0 for x in layer.DW[:, nodeNum]]
        if layer.DU is not None:
            layer.DU[:, nodeNum] = [0 for x in layer.DU[:, nodeNum]]
            if nodeNum < layer.DU.shape[0]:
                layer.DU[nodeNum, :] = [0 for x in layer.DU[nodeNum, :]]
        layer.DB[nodeNum] = 0

        if layer.next_layer is not None and isIntrinsicallyTrainableLayer(layer.next_layer):
            if nodeNum < layer.next_layer.DW.shape[0]:
                layer.next_layer.DW[nodeNum, :] = [0 for x in layer.next_layer.DW[nodeNum, :]]


def keepNode(layer, current_timestep=None, nodeNum=None):
    if layer.unrolled and current_timestep is not None:
        layer.DW[current_timestep][:, nodeNum] = layer.W[:, nodeNum]
        if layer.DU is not None:
            layer.DU[current_timestep][:, nodeNum] = layer.U[:, nodeNum]
            if nodeNum < layer.DU[current_timestep].shape[0]:
                layer.DU[current_timestep][nodeNum, :] = layer.U[nodeNum, :]

        layer.DB[current_timestep][nodeNum] = layer.B[nodeNum]

        if layer.next_layer is not None and isIntrinsicallyTrainableLayer(layer.next_layer):
            if layer.next_layer.unrolled:
                if nodeNum < layer.next_layer.DW[current_timestep].shape[0]:
                    layer.next_layer.DW[current_timestep][nodeNum, :] = layer.next_layer.W[nodeNum, :]
            elif current_timestep == layer.timestep - 1:
                if nodeNum < layer.next_layer.DW.shape[0]:
                    layer.next_layer.DW[nodeNum, :] = layer.next_layer.W[nodeNum, :]
    else:
        layer.DW[:, nodeNum] = layer.W[:, nodeNum]
        if layer.DU is not None:
            layer.DU[:, nodeNum] = layer.U[:, nodeNum]
            if nodeNum < layer.DU.shape[0]:
                layer.DU[nodeNum, :] = layer.U[nodeNum, :]
        layer.DB[nodeNum] = layer.B[nodeNum]

        if layer.next_layer is not None and isIntrinsicallyTrainableLayer(layer.next_layer):
            if nodeNum < layer.next_layer.DW.shape[0]:
                layer.next_layer.DW[nodeNum, :] = layer.next_layer.W[nodeNum, :]


def removeBottomKForPositiveExample(layer, timestep=None, removePercent=5, mode='active'):
    removePercent = removePercent / 100.0
    shouldReverse = True
    inactive_percent = {}
    for nodeNum in range(layer.num_node):
        if mode == 'active':

            if layer.unrolled:
                if (layer.inactive_count[timestep][nodeNum] + layer.active_count[timestep][nodeNum]) == 0:
                    inactivePercent = 0.0
                else:
                    inactivePercent = (layer.inactive_count[timestep][nodeNum] / (
                            layer.inactive_count[timestep][nodeNum] + layer.active_count[timestep][nodeNum])) * 100.0
            else:
                if (layer.inactive_count[nodeNum] + layer.active_count[nodeNum]) == 0:
                    inactivePercent = 0.0
                else:
                    inactivePercent = (layer.inactive_count[nodeNum] / (
                            layer.inactive_count[nodeNum] + layer.active_count[nodeNum])) * 100.0
        else:
            shouldReverse = False
            inactivePercent = (layer.node_sum[nodeNum] / (
                    layer.inactive_count[nodeNum] + layer.active_count[nodeNum]))

        inactive_percent[nodeNum] = inactivePercent

    inactive_percent = {k: v for k, v in
                        sorted(inactive_percent.items(), key=lambda item: item[1], reverse=shouldReverse)}
    removeUntil = removePercent * layer.num_node
    counter = 0
    for nodeNum in inactive_percent.keys():
        if counter <= removeUntil:
            removeNode(layer, timestep, nodeNum)
        else:
            keepNode(layer, timestep, nodeNum)
        counter += 1


def removeTopKForNegativeExample(layerPositive, layerNegative, timestep=None, removePercent=1, mode='active'):
    removePercent = removePercent / 100.0
    shouldReverse = False
    inactive_percent = {}
    for nodeNum in range(layerNegative.num_node):
        if mode == 'active':
            if layerNegative.unrolled:
                if (layerNegative.inactive_count[timestep][nodeNum] + layerNegative.active_count[timestep][
                    nodeNum]) == 0:
                    inactivePercent = 1.0
                else:
                    inactivePercent = (layerNegative.inactive_count[timestep][nodeNum] / (
                            layerNegative.inactive_count[timestep][nodeNum] + layerNegative.active_count[timestep][
                        nodeNum])) * 100.0
            else:
                if (layerNegative.inactive_count[nodeNum] + layerNegative.active_count[
                    nodeNum]) == 0:
                    inactivePercent = 1.0
                else:
                    inactivePercent = (layerNegative.inactive_count[nodeNum] / (
                            layerNegative.inactive_count[nodeNum] + layerNegative.active_count[nodeNum])) * 100.0
        else:
            shouldReverse = True
            inactivePercent = (layerNegative.node_sum[nodeNum] / (
                    layerNegative.inactive_count[nodeNum] + layerNegative.active_count[nodeNum]))
        inactive_percent[nodeNum] = inactivePercent

    inactive_percent = {k: v for k, v in
                        sorted(inactive_percent.items(), key=lambda item: item[1], reverse=shouldReverse)}
    removeUntil = removePercent * layerNegative.num_node
    counter = 0
    for nodeNum in inactive_percent.keys():
        if counter <= removeUntil:
            removeNode(layerPositive, timestep, nodeNum)
        counter += 1


def reAttachForPositiveExample(layer, timestep=None, keepPercent=20, mode='active'):
    keepPercent = keepPercent / 100.0
    shouldReverse = False
    inactive_percent = {}
    for nodeNum in range(layer.num_node):
        if mode == 'active':

            if layer.unrolled:
                if (layer.inactive_count[timestep][nodeNum] + layer.active_count[timestep][nodeNum]) == 0:
                    inactivePercent = 0.0
                else:
                    inactivePercent = (layer.inactive_count[timestep][nodeNum] / (
                            layer.inactive_count[timestep][nodeNum] + layer.active_count[timestep][nodeNum])) * 100.0
            else:
                if (layer.inactive_count[nodeNum] + layer.active_count[nodeNum]) == 0:
                    inactivePercent = 0.0
                else:
                    inactivePercent = (layer.inactive_count[nodeNum] / (
                            layer.inactive_count[nodeNum] + layer.active_count[nodeNum])) * 100.0
        else:
            shouldReverse = True
            inactivePercent = (layer.node_sum[nodeNum] / (
                    layer.inactive_count[nodeNum] + layer.active_count[nodeNum]))

        inactive_percent[nodeNum] = inactivePercent

    inactive_percent = {k: v for k, v in
                        sorted(inactive_percent.items(), key=lambda item: item[1], reverse=shouldReverse)}
    keepUntil = keepPercent * layer.num_node
    counter = 0
    for nodeNum in inactive_percent.keys():
        if counter <= keepUntil:
            keepNode(layer, timestep, nodeNum)
        counter += 1


def reAttachForNegativeExample(layerPositive, layerNegative, timestep=None, keepPercent=10, mode='active'):
    keepPercent = keepPercent / 100.0
    shouldReverse = True
    inactive_percent = {}
    for nodeNum in range(layerNegative.num_node):
        if mode == 'active':
            if layerNegative.unrolled:
                if (layerNegative.inactive_count[timestep][nodeNum] + layerNegative.active_count[timestep][
                    nodeNum]) == 0:
                    inactivePercent = 1.0
                else:
                    inactivePercent = (layerNegative.inactive_count[timestep][nodeNum] / (
                            layerNegative.inactive_count[timestep][nodeNum] + layerNegative.active_count[timestep][
                        nodeNum])) * 100.0
            else:
                if (layerNegative.inactive_count[nodeNum] + layerNegative.active_count[
                    nodeNum]) == 0:
                    inactivePercent = 1.0
                else:
                    inactivePercent = (layerNegative.inactive_count[nodeNum] / (
                            layerNegative.inactive_count[nodeNum] + layerNegative.active_count[nodeNum])) * 100.0
        else:
            shouldReverse = False
            inactivePercent = (layerNegative.node_sum[nodeNum] / (
                    layerNegative.inactive_count[nodeNum] + layerNegative.active_count[nodeNum]))
        inactive_percent[nodeNum] = inactivePercent

    inactive_percent = {k: v for k, v in
                        sorted(inactive_percent.items(), key=lambda item: item[1], reverse=shouldReverse)}
    keepUntil = keepPercent * layerNegative.num_node
    counter = 0
    for nodeNum in inactive_percent.keys():
        if counter <= keepUntil:
            keepNode(layerPositive, timestep, nodeNum)
        counter += 1


def removeConcernBasedOnActiveStat(layer, timestep=None, inactiveThreshold=1.0):
    for nodeNum in range(layer.num_node):
        if layer.unrolled:
            inactivePercent = (layer.inactive_count[timestep][nodeNum] / (
                    layer.inactive_count[timestep][nodeNum] + layer.active_count[timestep][nodeNum]))
        else:
            inactivePercent = (layer.inactive_count[nodeNum] / (
                    layer.inactive_count[nodeNum] + layer.active_count[nodeNum]))
        if inactivePercent >= inactiveThreshold:
            removeNode(layer, timestep, nodeNum)
        else:
            keepNode(layer, timestep, nodeNum)


def tangleConcernBasedOnActiveStat(layerPos, layerNeg, timestep=None, activeThreshold=1.0):
    for nodeNum in range(layerNeg.num_node):
        if layerNeg.unrolled:
            activePercent = (layerNeg.active_count[timestep][nodeNum] / (
                    layerNeg.inactive_count[timestep][nodeNum] + layerNeg.active_count[timestep][nodeNum])) * 100.0
        else:
            activePercent = (layerNeg.active_count[nodeNum] / (
                    layerNeg.inactive_count[nodeNum] + layerNeg.active_count[nodeNum]))

        if activePercent >= activeThreshold:
            keepNode(layerPos, timestep, nodeNum)


def removeConcernBasedOnMedian(layer, timestep=None, medianThreshold=0.1, maxRemove=0.3):
    d = {}
    removeCount = 0
    keepCount = 0
    for nodeNum in range(layer.num_node):
        removeFlag = False

        if timestep is not None:
            if math.fabs(layer.median_node_val[timestep][:, nodeNum]) <= medianThreshold:
                removeFlag = True
        elif math.fabs(layer.median_node_val[:, nodeNum]) <= medianThreshold:
            removeFlag = True

        if (keepCount + removeCount) > 0 and (removeCount / (keepCount + removeCount)) >= maxRemove:
            removeFlag = False

        if removeFlag:
            removeCount += 1
            removeNode(layer, timestep, nodeNum)
        else:
            keepCount += 1
            keepNode(layer, timestep, nodeNum)


def tangleConcernBasedOnMedian(layerPos, layerNeg, timestep=None, medianThreshold=0.95):
    d = {}
    for nodeNum in range(layerNeg.num_node):

        keepFlag = False

        if timestep is not None:
            if math.fabs(layerNeg.median_node_val[timestep][:, nodeNum]) >= medianThreshold:
                keepFlag = True
        elif math.fabs(layerNeg.median_node_val[:, nodeNum]) >= medianThreshold:
            keepFlag = True

        if keepFlag:
            keepNode(layerPos, timestep, nodeNum)


def removeAndTangleConcernBasedOnComparison(layerPos, layerNeg, timestep=None, tangleThreshold=0.0,
                                            maxRemove=0.1, atleastRemove=0.05):
    d = {}
    num_node = layerPos.num_node
    if ALL_GATE:
        if layerPos.type == LayerType.LSTM:
            num_node = num_node * 4
        if layerPos.type == LayerType.GRU:
            num_node = num_node * 3

    for nodeNum in range(num_node):
        if timestep is not None:
            d[nodeNum] = layerPos.median_node_val[timestep][:, nodeNum] - \
                         layerNeg.median_node_val[timestep][:, nodeNum]
        else:
            d[nodeNum] = layerPos.median_node_val[:, nodeNum] - layerNeg.median_node_val[:, nodeNum]

    removeCount = 0
    keepCount = 0

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1])}
    removedSet = set()
    first = True
    firstVal = None
    for nodeNum in d.keys():
        removeFlag = False
        if d[nodeNum] + tangleThreshold < 0.0:
            removeFlag = True

        if first:
            firstVal = d[nodeNum]

        if removeCount / num_node >= maxRemove:
            removeFlag = False

        if not first and d[nodeNum] + tangleThreshold < 0 and firstVal / d[nodeNum] < 1.1:
            removeFlag = True

        first = False
        if removeFlag:
            removeCount += 1
            removedSet.add(nodeNum)
            removeNode(layerPos, timestep, nodeNum)
            if not ALL_GATE and layerPos.type == LayerType.LSTM:
                removeNode(layerPos, timestep, nodeNum + num_node)
                removeNode(layerPos, timestep, nodeNum + (2 * num_node))
                removeNode(layerPos, timestep, nodeNum + (3 * num_node))
        else:
            keepCount += 1
            keepNode(layerPos, timestep, nodeNum)
            if not ALL_GATE and layerPos.type == LayerType.LSTM:
                keepNode(layerPos, timestep, nodeNum + num_node)
                keepNode(layerPos, timestep, nodeNum + (2 * num_node))
                keepNode(layerPos, timestep, nodeNum + (3 * num_node))


def removeAndTangleConcernBasedOnComparison(layerPos, layerNeg, timestep=None, tangleThreshold=0.0,
                                            maxRemove=0.1, atleastRemove=0.05):
    d = {}
    num_node = layerPos.num_node
    if ALL_GATE:
        if layerPos.type == LayerType.LSTM:
            num_node = num_node * 4
        if layerPos.type == LayerType.GRU:
            num_node = num_node * 3

    for nodeNum in range(num_node):
        if timestep is not None:
            d[nodeNum] = layerPos.median_node_val[timestep][:, nodeNum] - \
                         layerNeg.median_node_val[timestep][:, nodeNum]
        else:
            d[nodeNum] = layerPos.median_node_val[:, nodeNum] - layerNeg.median_node_val[:, nodeNum]

    removeCount = 0
    keepCount = 0

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1])}
    removedSet = set()
    first = True
    firstVal = None
    removedPerBlock = {}
    for nodeNum in d.keys():

        nodeBlock = int(nodeNum / layerPos.num_node)
        if nodeBlock not in removedPerBlock:
            removedPerBlock[nodeBlock] = 0

        removeFlag = False
        if d[nodeNum] + tangleThreshold < 0.0:
            removeFlag = True

        if first:
            firstVal = d[nodeNum]

        if not ALL_GATE and removeCount / num_node >= maxRemove:
            removeFlag = False

            if not first and d[nodeNum] + tangleThreshold < 0 and firstVal / d[nodeNum] < 1.1:
                removeFlag = True
        if ALL_GATE:
            if removedPerBlock[nodeBlock] / layerPos.num_node >= maxRemove:
                removeFlag = False

        first = False
        if removeFlag:
            removeCount += 1
            removedSet.add(nodeNum)
            removeNode(layerPos, timestep, nodeNum)
            removedPerBlock[nodeBlock] += 1
            if not ALL_GATE and layerPos.type == LayerType.LSTM:
                removeNode(layerPos, timestep, nodeNum + num_node)
                removeNode(layerPos, timestep, nodeNum + (2 * num_node))
                removeNode(layerPos, timestep, nodeNum + (3 * num_node))
            if not ALL_GATE and layerPos.type == LayerType.GRU:
                removeNode(layerPos, timestep, nodeNum + num_node)
                removeNode(layerPos, timestep, nodeNum + (2 * num_node))
        else:
            keepCount += 1
            keepNode(layerPos, timestep, nodeNum)
            if not ALL_GATE and layerPos.type == LayerType.LSTM:
                keepNode(layerPos, timestep, nodeNum + num_node)
                keepNode(layerPos, timestep, nodeNum + (2 * num_node))
                keepNode(layerPos, timestep, nodeNum + (3 * num_node))
            if not ALL_GATE and layerPos.type == LayerType.GRU:
                keepNode(layerPos, timestep, nodeNum + num_node)
                keepNode(layerPos, timestep, nodeNum + (2 * num_node))


def tangleConcernBasedOnComparison(layerPos, layerNeg, timestep=None, atleastRemove=0.05, keepPercent=0.2):
    d = {}
    removeCount = 0
    keepCount = 0
    for nodeNum in range(layerNeg.num_node):

        if timestep is not None:
            d[nodeNum] = layerNeg.median_node_val[timestep][:, nodeNum]
        else:
            d[nodeNum] = layerNeg.median_node_val[:, nodeNum]

        if isNodeRemoved(layerPos, timestep, nodeNum):
            removeCount += 1
        else:
            keepCount += 1

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1], reverse=True)}
    keepUntil = keepPercent * layerNeg.num_node
    counter = 0
    for nodeNum in d.keys():

        if counter <= keepUntil:
            if isNodeRemoved(layerPos, timestep, nodeNum):
                tRc = removeCount - (counter + 1)
                tKc = keepCount + (counter + 1)
                if (tRc + tKc) > 0 and (tRc / (tKc + tRc)) >= atleastRemove:
                    keepNode(layerPos, timestep, nodeNum)
                    counter += 1


def removeAndTangleConcernBasedActiveStat(layerPos, layerNeg, timestep=None, tangleThreshold=-1.0,
                                          maxRemove=0.2):
    d = {}
    num_node = layerPos.num_node

    for nodeNum in range(num_node):
        if timestep is not None:
            d[nodeNum] = layerPos.median_node_val[timestep][:, nodeNum] - \
                         layerNeg.median_node_val[timestep][:, nodeNum]
        else:
            d[nodeNum] = layerPos.median_node_val[:, nodeNum] - layerNeg.median_node_val[:, nodeNum]

    removeCount = 0
    keepCount = 0

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1])}
    removedSet = set()
    for nodeNum in d.keys():

        removeFlag = False
        if d[nodeNum] <= tangleThreshold:
            removeFlag = True

        if removeCount / layerPos.num_node >= maxRemove:
            removeFlag = False

        if removeFlag:
            removeCount += 1
            removedSet.add(nodeNum)
            removeNode(layerPos, timestep, nodeNum)

        else:
            keepCount += 1
            keepNode(layerPos, timestep, nodeNum)
