#include "face_tensor_extractor.h"


namespace extractornamespace {


/* For CenterNetFacedetection */
//extern "C"
static std::vector<int> getIds(float *heatmap, int h, int w, float thresh)
{
	std::vector<int> ids;
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{

			//			std::cout<<"ids"<<heatmap[i*w+j]<<std::endl;
			if (heatmap[i * w + j] > thresh)
			{
				//				std::array<int, 2> id = { i,j };
				ids.push_back(i);
				ids.push_back(j);
				//	std::cout<<"print ids"<<i<<std::endl;
			}
		}
	}
	return ids;
}

static void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, float nmsthreshold)
{
	std::sort(input.begin(), input.end(),
			  [](const FaceInfo &a, const FaceInfo &b) {
				  return a.score > b.score;
			  });

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for (int i = 0; i < box_num; i++)
	{
		if (merged[i])
			continue;

		output.push_back(input[i]);

		float h0 = input[i].y2 - input[i].y1 + 1;
		float w0 = input[i].x2 - input[i].x1 + 1;

		float area0 = h0 * w0;

		for (int j = i + 1; j < box_num; j++)
		{
			if (merged[j])
				continue;

			float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1; //std::max(input[i].x1, input[j].x1);
			float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

			float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2; //bug fixed ,sorry
			float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if (inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h1 = input[j].y2 - input[j].y1 + 1;
			float w1 = input[j].x2 - input[j].x1 + 1;

			float area1 = h1 * w1;

			float score;

			score = inner_area / (area0 + area1 - inner_area);

			if (score > nmsthreshold)
				merged[j] = 1;
		}
	}
}
class Extractor::Impl {
public:
	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res);
	// cv::Mat AlignPlate(const cv::Mat& dst, const cv::Mat& src);


private:
    
    // bool cmp(FaceInfo& a, FaceInfo& b);
    float iou(float lbox[4], float rbox[4]);
    void nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height);
    void create_anchor_retinaface(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height);
    
};

Extractor::Extractor() {
    impl_ = new Impl();
}

Extractor::~Extractor() {
    if (impl_) {
        delete impl_;
    }
}

void Extractor::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
    return impl_->facelmks(l_user, res);
}

void Extractor::Impl::CenterFacelmks(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, 
               NvDsInferNetworkInfo &network_info, std::vector<FaceInfo>& res){
            //Change the Extractor Using CenterNet

        auto layerFinder = [&outputLayersInfo](const std::string &name)
		-> const NvDsInferLayerInfo * {
		for (auto &layer : outputLayersInfo)
		{

			if (layer.dataType == FLOAT &&
				(layer.layerName && name == layer.layerName))
			{
				return &layer;
			}
		}
		return nullptr;
	};
	const NvDsInferLayerInfo *heatmap = layerFinder("537");
	const NvDsInferLayerInfo *scale = layerFinder("538");
	const NvDsInferLayerInfo *offset = layerFinder("539");
	const NvDsInferLayerInfo *landmarks = layerFinder("540");
    if (!heatmap || !scale || !offset || !landmarks)
	{
		std::cerr << "ERROR: some layers missing or unsupported data types "
				  << "in output tensors" << std::endl;
		return false;
	}

	int fea_h = heatmap->inferDims.d[1];//; //#heatmap.size[2];
	int fea_w = heatmap->inferDims.d[2];//;//heatmap.size[3];
	int spacial_size = fea_h* fea_w;
 	// std::cout<<"features"<<fea_h<<"width"<<fea_w<<std::endl;
	// std::cout<<"0:"<<heatmap->inferDims.d[0]<<"1:"<<heatmap->inferDims.d[1]<<std::endl;
	float *heatmap_ = (float *)(heatmap->buffer);

	float *scale0 = (float *)(scale->buffer);
	float *scale1 = scale0 + spacial_size;

	float *offset0 = (float *)(offset->buffer);
	float *offset1 = offset0 + spacial_size;
	float *lm = (float *)landmarks->buffer;

	float scoreThresh = 0.5;
	std::vector<int> ids = getIds(heatmap_, fea_h, fea_w, scoreThresh);
	//?? d_w, d_h
	//std::cerr<<"getids:"<<ids.size()<<std::endl;
	int width = networkInfo.width;
	int height = networkInfo.height; 
	int d_h = (int)(std::ceil(height / 32) * 32);
	int d_w = (int)(std::ceil(width / 32) * 32); 
    std::vector<FaceInfo> tmpVec;
    for (int i = 0; i < ids.size() / 2; i++)
	{
		int id_h = ids[2 * i];
		int id_w = ids[2 * i + 1];
		int index = id_h * fea_w + id_w;

		float s0 = std::exp(scale0[index]) * 4;
		float s1 = std::exp(scale1[index]) * 4;
		float o0 = offset0[index];
		float o1 = offset1[index];
		float x1 = std::max(0., (id_w + o1 + 0.5) * 4 - s1 / 2);
		float y1 = std::max(0., (id_h + o0 + 0.5) * 4 - s0 / 2);
		float x2 = 0, y2 = 0;
		x1 = std::min(x1, (float)d_w);
		y1 = std::min(y1, (float)d_h);
		x2 = std::min(x1 + s1, (float)d_w);
		y2 = std::min(y1 + s0, (float)d_h);

		FaceInfo facebox;
		facebox.bbox[0] = x1;
		facebox.bbox[1] = y1;
		facebox.bbox[2] = x2;
		facebox.bbox[3] = y2;
		facebox.score = heatmap_[index];
		for (int j = 0; j < 5; j++)
		{
			facebox.anchor[2 * j] = x1 + lm[(2 * j + 1) * spacial_size + index] * s1;
			facebox.anchor[2 * j + 1] = y1 + lm[(2 * j) * spacial_size + index] * s0;
		}
		//std::cout<<"face:"<<x1<<","<<y1<<"---------"<<x2<<","<<y2<<std::endl;
		tmpVec.emplace_back(std::move(facebox));
	}
    const float threshold = 0.4;
	nms(tmpVec, res, threshold);
    
}

void Extractor::Impl::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
    static guint use_device_mem = 0;
    for (;l_user != NULL; l_user = l_user->next) { 
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
        continue; 
        }
        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        CenterFacelmks(meta->output_layers_info, meta->network_info, res);

        NvDsInferLayerInfo *info = &meta->output_layers_info[0];
        info->buffer = meta->out_buf_ptrs_host[0];
        if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
        // get all data from gpu to cpu
        cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
            info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
        std::vector < NvDsInferLayerInfo > outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
        float *output = (float*)(outputLayersInfo[0].buffer);
        std::vector<FaceInfo> temp;
        // std::vector<Detection> res;
        create_anchor_retinaface(temp, output, CONF_THRESH, NETWIDTH, NETHEIGHT);
        nms_and_adapt(temp, res, NMS_THRESH, NETWIDTH, NETHEIGHT);
    }  
}

bool cmp(FaceInfo& a, FaceInfo& b) {
    return a.score > b.score;
}

float Extractor::Impl::iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

void Extractor::Impl::nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), cmp);
    for (unsigned int m = 0; m < det.size(); ++m) {
        auto& item = det[m];
        res.push_back(item);
        for (unsigned int n = m + 1; n < det.size(); ++n) {
            if (iou(item.bbox, det[n].bbox) > nms_thresh) {
                det.erase(det.begin()+n);
                --n;
            }
        }
    }
    // crop larger area for better alignment performance 
    // there I choose to crop 50 more pixel 
    for (unsigned int m = 0; m < res.size(); ++m) {
        res[m].bbox[0] = CLIP(res[m].bbox[0]-25, 0, width - 1);
        res[m].bbox[1] = CLIP(res[m].bbox[1]-25 , 0, height -1);
        res[m].bbox[2] = CLIP(res[m].bbox[2]+25, 0, width - 1);
        res[m].bbox[3] = CLIP(res[m].bbox[3]+25, 0, height - 1);
    }

}

void Extractor::Impl::create_anchor_retinaface(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(FaceInfo) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        
        FaceInfo det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        // det
        res.push_back(det);
        
    }
}


}
