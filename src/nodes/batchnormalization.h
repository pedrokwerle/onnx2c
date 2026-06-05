/* This file is part of onnx2c.
 *
 * BatchNormalization.
 * Algorithm as "described in this paper https://arxiv.org/abs/1502.03167"
 *
 * i.e. This operator provides a "whitening" of the data in the middle of
 * the network, (as opposed to just a preprocessing the input).
 *
 * Algorithm calculated is:
 *
 * 	y = scale * X + bias
 * where
 * 	X = (x - mean) / sqrt( variance + epsilon )
 *
 *
 * mean and variance can, optionally, be updated and sent as output,
 * but this is still unimplemented in onnx2c.
 */
namespace toC {

class BatchNormalization : public Node {
	public:
	BatchNormalization()
	{
		op_name = "BatchNormalization";
		epsilon = 1e-5;
		momentum = 0.9;
		sqrt_var_offline = false;
	}
	bool sqrt_var_offline; // TODO: is it ever possible that we can't compute sqrt(var) offline?

	float epsilon;
	float momentum;

	void parseAttribute_epsilon(const onnx::AttributeProto& a)
	{
		if (a.type() != onnx::AttributeProto_AttributeType_FLOAT)
			ERROR("Bad attribute " << a.name());
		if (a.has_f() == false)
			ERROR("Bad attribute " << a.name());
		epsilon = a.f();
	}
	void parseAttribute_momentum(const onnx::AttributeProto& a)
	{
		if (a.type() != onnx::AttributeProto_AttributeType_FLOAT)
			ERROR("Bad attribute " << a.name());
		if (a.has_f() == false)
			ERROR("Bad attribute " << a.name());
		momentum = a.f();
	}

	virtual void parseAttributes(onnx::NodeProto& node) override
	{

		for (const auto& a : node.attribute()) {
			if (a.name() == "epsilon")
				parseAttribute_epsilon(a);
			else if (a.name() == "momentum")
				parseAttribute_momentum(a);
			else if (a.name() == "spatial") {
				// NB: spatial was removed in onnx opset v. 9.
				int spatial = parse_attribute_int(a);
				if (spatial != 1)
					ERROR("non-default value for 'spatial' attribute not implemented");
			}
			else if (a.name() == "training_mode") {
				// Do nothing
				LOG(TRACE) << " Ignoring attribute '" << a.name() << "'" << std::endl;
			}
			else
				ERROR("Unknown attribute " << a.name());
		}
	}

	virtual void print(std::ostream& dst) const override
	{
		const Tensor* input = get_input_tensor(0);
		const Tensor* scale = get_input_tensor(1);
		const Tensor* bias = get_input_tensor(2);
		int batch_size = input->data_dim[0];
		int num_chan = input->data_dim[1];
		std::string type = input->data_type_str();

		dst << "\t/* BatchNormalization" << std::endl;
		dst << "\t * epsilon = " << epsilon << std::endl;
		dst << "\t * momentum = " << momentum << std::endl;
		dst << "\t */" << std::endl
		    << std::endl;

		if (sqrt_var_offline == false)
			INDT_1 << "float epsilon = " << epsilon << ";" << std::endl;

		dst << "\t" << "for( int32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		dst << "\t" << "for( int32_t c=0; c<" << num_chan << "; c++ ) {" << std::endl;

		// create the indexing string for picking out an element in input/output
		std::string idxs = "[b][c]";
		for (unsigned i = 2; i < input->data_dim.size(); i++)
			idxs += "[i" + std::to_string(i) + "]";

		// Loop over data dimensions
		for (unsigned i = 2; i < input->data_dim.size(); i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst << idx << "<" << input->data_dim[i] << "; ";
			dst << idx << "++ ) {" << std::endl;
		}

		INDT_2 << type << " tmp_X = ( X" << idxs << " - mean[c] ) / ";
		if (sqrt_var_offline)
			dst << "( var[c] );" << std::endl;
		else
			dst << "( sqrt( var[c] + epsilon));" << std::endl;

		INDT_2 << "output" << idxs << " = tmp_X ";

		if (!isSplatted(scale, 1.0f))
			dst << "* scale[c]";
		if (!isSplatted(bias, 0.0f))
			dst << " + bias[c]";
		dst << ";" << std::endl;

		for (unsigned i = 2; i < input->data_dim.size(); i++)
			dst << "\t}" << std::endl;

		dst << "\t" << "}" << std::endl;
		dst << "\t" << "}" << std::endl;
	}

	// TODO: this could be useful elsewhere too
	bool isSplatted(const Tensor* t, float value) const
	{
		if (t->isConst == false)
			return false;

		if (t->data_type == onnx::TensorProto_DataType_FLOAT) {
			float* b = (float*)t->data_buffer;
			for (int i = 0; i < t->data_num_elem(); i++)
				if (b[i] != value)
					return false;
		} else if (t->data_type == onnx::TensorProto_DataType_FLOAT16) {
			_Float16* b = (_Float16*)t->data_buffer;
			_Float16 val16 = static_cast<_Float16>(value);
			for (int i = 0; i < t->data_num_elem(); i++)
				if (b[i] != val16)
					return false;
		} else {
			ERROR("isSplatted: unsupported data type");
		}

		return true;
	}

	// Updates variance tensor in-place to contain the entire denominator
	// of the BatchNormalization formula.
	// TODO: This breaks if var is used anywere else.
	void calculateSqrtVarOffline(const Tensor* var)
	{
		LOG(TRACE) << "Test 8" << std::endl;
		if (var->data_type == onnx::TensorProto_DataType_FLOAT) {
			float* v = (float*)var->data_buffer;
			LOG(TRACE) << "Test 9 (FP32)" << std::endl;
			for (int i = 0; i < var->data_num_elem(); i++) {
				LOG(TRACE) << "Loop " << std::to_string(i) << std::endl;
				LOG(TRACE) << "v[i] = " << std::to_string(v[i]) << " epsilon = " << std::to_string(epsilon) << std::endl;
				v[i] = sqrt(v[i] + epsilon);
			}
		} else if (var->data_type == onnx::TensorProto_DataType_FLOAT16) {
			_Float16* v = (_Float16*)var->data_buffer;
			LOG(TRACE) << "Test 9 (FP16)" << std::endl;
			for (int i = 0; i < var->data_num_elem(); i++) {
				LOG(TRACE) << "Loop " << std::to_string(i) << std::endl;
				float val = static_cast<float>(v[i]);
				LOG(TRACE) << "v[i] = " << std::to_string(val) << " epsilon = " << std::to_string(epsilon) << std::endl;
				v[i] = static_cast<_Float16>(sqrt(val + epsilon));
			}
		} else {
			ERROR("calculateSqrtVarOffline: unsupported data type");
		}
		LOG(TRACE) << "Test 10" << std::endl;
	}

	virtual void resolve(void) override
	{
		LOG(TRACE) << "Test 0" << std::endl;
		if (get_number_of_inputs() != 5)
			ERROR("wrong number of inputs to BatchNormalization");
		LOG(TRACE) << "Test 1" << std::endl;
		name_input(0, "X");
		LOG(TRACE) << "Test 2" << std::endl;
		name_input(1, "scale");
		LOG(TRACE) << "Test 3" << std::endl;
		name_input(2, "bias");
		LOG(TRACE) << "Test 4" << std::endl;
		name_input(3, "mean");
		LOG(TRACE) << "Test 5" << std::endl;
		name_input(4, "var");
		LOG(TRACE) << "Test 6" << std::endl;

		if (get_input_tensor(4)->isConst) {
			LOG(TRACE) << "Test 7" << std::endl;
			calculateSqrtVarOffline(get_input_tensor(4));
			LOG(TRACE) << "Test 8" << std::endl;
			sqrt_var_offline = true;
			LOG(TRACE) << "Test 9" << std::endl;
		}

		Tensor* rv = new Tensor;
		rv->data_dim = get_input_tensor(0)->data_dim;
		rv->data_type = get_input_tensor(0)->data_type;
		register_output(rv, "output");
	}
};
} // namespace toC