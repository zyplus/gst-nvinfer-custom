#include "tensor_extractor.h"


namespace extractornamespace {
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
class Extractor::Impl {
public:
	void facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res);
    bool platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res);
	// cv::Mat AlignPlate(const cv::Mat& dst, const cv::Mat& src);


private:
    
    // bool cmp(FaceInfo& a, FaceInfo& b);
    // bool cmp(PlateInfo& a, PlateInfo& b);
    float iou(float lbox[4], float rbox[4]);
    void nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height);
    bool nms_and_adapt_plate(std::vector<PlateInfo>& det, std::vector<PlateInfo>& res, float nms_thresh, int width, int height);

    void decode_bbox_retina_face(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height);
    void decode_bbox_retina_plate(std::vector<anchorBox> &anchor, std::vector<PlateInfo>& res, float *bbox, float *lmk, float *conf, 
                                float bbox_threshold, int width, int height);
    void create_anchor_retina_plate(std::vector<anchorBox> &anchor, int w, int h);
    void CenterFacelmks(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
										 NvDsInferNetworkInfo &network_info, std::vector<FaceInfo> &res);
    
};

Extractor::Extractor() {
    impl_ = new Impl();
}

Extractor::~Extractor() {
    if (impl_) {
        delete impl_;
    }
}

	void Extractor::facelmks(NvDsMetaList *l_user, std::vector<FaceInfo> &res)
	{
		return impl_->facelmks(l_user, res);
	}

	void Extractor::Impl::CenterFacelmks(std::vector<NvDsInferLayerInfo > const &outputLayersInfo,
										 NvDsInferNetworkInfo &network_info, std::vector<FaceInfo> &res)
	{
		// Change the Extractor Using CenterNet
 
		auto layerFinder = [&outputLayersInfo](const std::string &name)
			-> const NvDsInferLayerInfo *
		{
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
			std::cout << "ERROR: some layers missing or unsupported data types "
					  << "in output tensors" << std::endl;
			return ;
		}

		int fea_h = heatmap->inferDims.d[1]; //; //#heatmap.size[2];
		int fea_w = heatmap->inferDims.d[2]; //;//heatmap.size[3];
		int spacial_size = fea_h * fea_w;
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
		std::cout<<"getids:"<<ids.size()<<std::endl;
		int width = network_info.width;
		int height = network_info.height;
		int d_h = (int)(std::ceil(height / 32) * 32);
		int d_w = (int)(std::ceil(width / 32) * 32);
		// std::vector<FaceInfo> tmpVec;
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
			facebox.bbox[0] = (x1+x2) / 2;
			facebox.bbox[1] = (y1+y2) / 2;
			facebox.bbox[2] = x2-x1;
			facebox.bbox[3] = y2-y1;
			facebox.confidence = heatmap_[index];
			for (int j = 0; j < 5; j++)
			{
				facebox.landmark[2 * j] = x1 + lm[(2 * j + 1) * spacial_size + index] * s1;
				facebox.landmark[2 * j + 1] = y1 + lm[(2 * j) * spacial_size + index] * s0;
			}
			// std::cout<<"face:"<<x1<<","<<y1<<"---------"<<x2<<","<<y2<<std::endl;
			res.emplace_back(std::move(facebox));
		}
		// const float threshold = 0.4;
		// nms(tmpVec, res, threshold);
	}

// void Extractor::Impl::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
//     static guint use_device_mem = 0;
//     for (;l_user != NULL; l_user = l_user->next) { 
//         NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
//         if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
//         continue; 
//         }
//         /* convert to tensor metadata */
//         NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
//         NvDsInferLayerInfo *info = &meta->output_layers_info[0];
//         info->buffer = meta->out_buf_ptrs_host[0];
//         if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
//         // get all data from gpu to cpu
//         cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
//             info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
//         }
//         std::vector < NvDsInferLayerInfo > outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
//         float *output = (float*)(outputLayersInfo[0].buffer);
//         std::vector<FaceInfo> temp;
//         decode_bbox_retina_face(temp, output, CONF_THRESH, FACE_NETWIDTH, FACE_NETHEIGHT);
//         nms_and_adapt(temp, res, NMS_THRESH, FACE_NETWIDTH, FACE_NETHEIGHT);
//     }  
// }


	void Extractor::Impl::facelmks(NvDsMetaList *l_user, std::vector<FaceInfo> &res)
	{
		static guint use_device_mem = 0;
		for (; l_user != NULL; l_user = l_user->next)
		{
			NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
			if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
			{
				continue;
			}
			/* convert to tensor metadata */
			NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
			std::vector<FaceInfo> temp;
			
			for(int i =0; i < meta->num_output_layers; ++i){
				 NvDsInferLayerInfo *info = &meta->output_layers_info[i];
				 info->buffer = meta->out_buf_ptrs_host[i];
				// std::cout<<"device is " <<meta->out_buf_ptrs_dev[i]<<std::endl;
				// if(meta->out_buf_ptrs_dev[i]){
				// 	cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                //   info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);

				// }
			}
			std::vector<NvDsInferLayerInfo > layerInfoVec(meta->output_layers_info,
                            meta->output_layers_info + meta->num_output_layers);
			CenterFacelmks(layerInfoVec, meta->network_info, temp);
			std::cout<<"get temp count "<<temp.size()<<std::endl;
			nms_and_adapt(temp, res, NMS_THRESH, meta->network_info.width, meta->network_info.height);
			std::cout<<"get res count "<<res.size()<<std::endl;
			
		}
	}

	bool cmp(FaceInfo &a, FaceInfo &b)
	{
		return a.confidence > b.confidence;
	}

	float Extractor::Impl::iou(float lbox[4], float rbox[4])
	{
		float interBox[] = {
			std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
			std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
			std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
			std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
		};

		if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
			return 0.0f;

		float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
		return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
	}

	void Extractor::Impl::nms_and_adapt(std::vector<FaceInfo> &det, std::vector<FaceInfo> &res, float nms_thresh, int width, int height)
	{
		std::sort(det.begin(), det.end(), cmp);
		for (unsigned int m = 0; m < det.size(); ++m)
		{
			auto &item = det[m];
			res.push_back(item);
			for (unsigned int n = m + 1; n < det.size(); ++n)
			{
				if (iou(item.bbox, det[n].bbox) > nms_thresh)
				{
					det.erase(det.begin() + n);
					--n;
				}
			}
		}
		// crop larger area for better alignment performance
		// there I choose to crop 50 more pixel
		// std::cout<<"res 1 count " << res.size()<<std::endl;
		for (unsigned int m = 0; m < res.size(); ++m)
		{
			float wide_dist = pow((res[m].landmark[0] - res[m].landmark[2]) * (res[m].landmark[0] - res[m].landmark[2]) + \
			                 (res[m].landmark[1] - res[m].landmark[3]) * (res[m].landmark[1] - res[m].landmark[3]), 0.5);
			float high_dist = pow((res[m].landmark[0] - res[m].landmark[6]) * (res[m].landmark[0] - res[m].landmark[6]) + \
			                  (res[m].landmark[1] - res[m].landmark[7]) * (res[m].landmark[1] - res[m].landmark[7]), 0.5);
			float dist_rate = high_dist / wide_dist;

			float dist_C = pow((res[m].landmark[6] - res[m].landmark[4]) * (res[m].landmark[6] - res[m].landmark[4]) + \
			               (res[m].landmark[7] - res[m].landmark[5]) * (res[m].landmark[7] - res[m].landmark[5]), 0.5);
			float dist_D = pow((res[m].landmark[8] - res[m].landmark[4]) * (res[m].landmark[8] - res[m].landmark[4]) + \
			                 (res[m].landmark[9] - res[m].landmark[5]) * (res[m].landmark[9] - res[m].landmark[5]), 0.5);
			float width_rate = fabs(dist_C / dist_D - 1);

			if ( !(dist_rate < 1.5 && width_rate < 0.7) ){
				res.erase(res.begin()+m);
				m--;
				continue;

			}
			res[m].bbox[0] = (int)CLIP(res[m].bbox[0] - res[m].bbox[2] / 2 - 10, 0, width - 1);
			res[m].bbox[1] = (int)CLIP(res[m].bbox[1] - res[m].bbox[3] / 2 - 10, 0, height - 1);
			res[m].bbox[2] = (int)CLIP(res[m].bbox[0] + res[m].bbox[2] + 20, 0, width - 1);
			res[m].bbox[3] = (int)CLIP(res[m].bbox[1] + res[m].bbox[3] + 20, 0, height - 1);
		}
        // std::cout<<"get res count is " <<res.size()<<std::endl;
	}


// void Extractor::facelmks(NvDsMetaList * l_user, std::vector<FaceInfo>& res) {
//     return impl_->facelmks(l_user, res);
// }

bool Extractor::platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res) {
    return impl_->platelmks(l_user, res);
}


bool Extractor::Impl::platelmks(NvDsMetaList * l_user, std::vector<PlateInfo>& res) {
    static guint use_device_mem = 1;
    bool flag = false;
    for (;l_user != NULL; l_user = l_user->next) { 
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
            continue; 
        }
        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        NvDsInferLayerInfo *info = &meta->output_layers_info[0];
        info->buffer = meta->out_buf_ptrs_host[0];
        if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
            // get all data from gpu to cpu
            cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
                info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
        std::vector<NvDsInferLayerInfo> outputLayersInfo (meta->output_layers_info, meta->output_layers_info + meta->num_output_layers);
        float *bbox = (float*)(outputLayersInfo[0].buffer);
        float *lmks = (float*)(outputLayersInfo[1].buffer);
        float *conf = (float*)(outputLayersInfo[2].buffer);

        std::vector<anchorBox> anchor;
        std::vector<PlateInfo> temp;
        create_anchor_retina_plate(anchor, PLATE_NETWIDTH, PLATE_NETHEIGHT);
        decode_bbox_retina_plate(anchor, temp, bbox, lmks, conf, CONF_THRESH, PLATE_NETWIDTH, PLATE_NETHEIGHT);
        flag = nms_and_adapt_plate(temp, res, NMS_THRESH, PLATE_NETWIDTH, PLATE_NETHEIGHT);
    }  
    return flag;
}

// float Extractor::Impl::iou(float lbox[4], float rbox[4]) {
//     float interBox[] = {
//         std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
//         std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
//         std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
//         std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
//     };

//     if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
//         return 0.0f;

//     float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
//     return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
// }

// void Extractor::Impl::nms_and_adapt(std::vector<FaceInfo>& det, std::vector<FaceInfo>& res, float nms_thresh, int width, int height) {
//     std::sort(det.begin(), det.end(), [](FaceInfo& a, FaceInfo& b){return a.confidence > b.confidence;});
//     for (unsigned int m = 0; m < det.size(); ++m) {
//         auto& item = det[m];
//         res.push_back(item);
//         for (unsigned int n = m + 1; n < det.size(); ++n) {
//             if (iou(item.bbox, det[n].bbox) > nms_thresh) {
//                 det.erase(det.begin()+n);
//                 --n;
//             }
//         }
//     }
//     // crop larger area for better alignment performance 
//     // there I choose to crop 50 more pixel 
//     for (unsigned int m = 0; m < res.size(); ++m) {
//         res[m].bbox[0] = CLIP(res[m].bbox[0]-10, 0, width - 1);
//         res[m].bbox[1] = CLIP(res[m].bbox[1]-10, 0, height -1);
//         res[m].bbox[2] = CLIP(res[m].bbox[2]+20, 0, width - 1);
//         res[m].bbox[3] = CLIP(res[m].bbox[3]+20, 0, height - 1);
//     }

// }

bool Extractor::Impl::nms_and_adapt_plate(std::vector<PlateInfo>& det, std::vector<PlateInfo>& res, float nms_thresh, int width, int height) {
    std::sort(det.begin(), det.end(), [](PlateInfo& a, PlateInfo& b){return a.confidence > b.confidence;});
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
    // std::cout<<"after nms, size: "<<res.size()<<std::endl;
    // top k
    std::sort(res.begin(), res.end(), [](PlateInfo& a, PlateInfo& b){return a.confidence > b.confidence;});
    if(res.size() > 1){
        res.erase(res.begin()+1, res.end()); 

    }
    // std::cout<<"after topk, size: "<<res.size()<<std::endl;
    // if nothing extracted, return false
    if(res.size() != 0){
        return true;
    }
    else{
        return false;
    }
}

void Extractor::Impl::decode_bbox_retina_face(std::vector<FaceInfo>& res, float *output, float conf_thresh, int width, int height) {
    int det_size = sizeof(FaceInfo) / sizeof(float);
    for (int i = 0; i < output[0]; i++){
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        FaceInfo det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
        det.bbox[1] = CLIP(det.bbox[1] , 0, height -1);
        det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
        det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
        res.push_back(det);
        
    }
}

void Extractor::Impl::create_anchor_retina_plate(std::vector<anchorBox> &anchor, int w, int h) {
    anchor.clear();
    std::vector<std::vector<int>> feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (unsigned int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h / steps[i]));
        feature_map[i].push_back(ceil(w / steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (unsigned int k = 0; k < feature_map.size(); ++k) {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i) {
            for (int j = 0; j < feature_map[k][1]; ++j) {
                for (unsigned int l = 0; l < min_size.size(); ++l) {
                    float s_kx = min_size[l] * 1.0 / w;
                    float s_ky = min_size[l] * 1.0 / h;
                    float cx = (j + 0.5) * steps[k] / w;
                    float cy = (i + 0.5) * steps[k] / h;
                    anchorBox axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }
    }
}

void Extractor::Impl::decode_bbox_retina_plate(std::vector<anchorBox> &anchor, std::vector<PlateInfo>& res, float *bbox, float *lmk, float *conf, 
                                float bbox_threshold, int width, int height) {
    for (unsigned int i = 0; i < anchor.size(); ++i) {
        if (*(conf + 1) > bbox_threshold) {
            anchorBox tmp = anchor[i];
            anchorBox tmp1;
        
            // decode bbox
            // std::cout<<tmp.cx<<" "<<tmp.cy<<" "<<tmp.sx<<" "<<tmp.sy<<std::endl;
            tmp1.cx = tmp.cx + *bbox * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(bbox + 1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(bbox + 2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(bbox + 3) * 0.2);

            PlateInfo det;
            det.bbox[0] = (tmp1.cx - tmp1.sx / 2) * width;
            det.bbox[1] = (tmp1.cy - tmp1.sy / 2) * height;
            det.bbox[2] = (tmp1.cx + tmp1.sx / 2) * width - det.bbox[0];
            det.bbox[3] = (tmp1.cy + tmp1.sy / 2) * height - det.bbox[1];

            det.bbox[0] = CLIP(det.bbox[0], 0, width - 1);
            det.bbox[1] = CLIP(det.bbox[1], 0, height -1); 
            det.bbox[2] = CLIP(det.bbox[2], 0, width - 1);
            det.bbox[3] = CLIP(det.bbox[3], 0, height - 1);
  
            det.confidence = *(conf + 1);
            
            for(unsigned int j = 0; j < 8; ){
                
                det.landmark[j]   = (tmp.cx + *(lmk + j) * 0.1 * tmp.sx) * width;
                det.landmark[j+1] = (tmp.cy + *(lmk + j + 1) * 0.1 * tmp.sy) * height;
                j = j + 2;
            }
            res.push_back(det);
        }
        
        bbox += 4;
        conf += 2;
        lmk  += 8;
        
    }
}

}
