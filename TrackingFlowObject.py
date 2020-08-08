class TrackingFlowObject(object):
    def __init__(self, height, width, flows, fps, framecount):
        self.input_height = height
        self.input_width = width
        self.flows = flows
        self.fps = fps
        self.framecount = framecount
