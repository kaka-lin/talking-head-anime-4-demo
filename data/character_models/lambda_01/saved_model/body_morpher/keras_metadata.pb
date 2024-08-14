
�root"_tf_keras_model*�{"name": "siren_morpher03", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SirenMorpher03", "config": {}, "shared_object_id": 0, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 4]}, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 512, 512, 4]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 45]}, "float32", null]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 512, 512, 4]}, "float32", "input_1"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "SirenMorpher03", "config": {}}}2
�		root.last_linear"_tf_keras_layer*�	{"name": "last_linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "last_linear", "trainable": true, "dtype": "float32", "filters": 7, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 90}}, "shared_object_id": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 90]}}2
�(root.siren_layers.0"_tf_keras_sequential*�{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 47]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.0.0_input"}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.0", "in_channels": 47, "out_channels": 360, "is_first": true}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.1", "in_channels": 360, "out_channels": 360, "is_first": false}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.2", "in_channels": 360, "out_channels": 180, "is_first": false}}]}, "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 47]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 47]}, "float32", "siren_layers.0.0_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 128, 128, 47]}, "float32", "siren_layers.0.0_input"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 47]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.0.0_input"}, "shared_object_id": 5}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.0", "in_channels": 47, "out_channels": 360, "is_first": true}, "shared_object_id": 6}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.1", "in_channels": 360, "out_channels": 360, "is_first": false}, "shared_object_id": 7}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.2", "in_channels": 360, "out_channels": 180, "is_first": false}, "shared_object_id": 8}]}}}2
�)root.siren_layers.1"_tf_keras_sequential*�{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 227]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.1.0_input"}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.0", "in_channels": 227, "out_channels": 180, "is_first": false}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.1", "in_channels": 180, "out_channels": 180, "is_first": false}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.2", "in_channels": 180, "out_channels": 90, "is_first": false}}]}, "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 227]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 256, 256, 227]}, "float32", "siren_layers.1.0_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 256, 256, 227]}, "float32", "siren_layers.1.0_input"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 227]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.1.0_input"}, "shared_object_id": 10}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.0", "in_channels": 227, "out_channels": 180, "is_first": false}, "shared_object_id": 11}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.1", "in_channels": 180, "out_channels": 180, "is_first": false}, "shared_object_id": 12}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.2", "in_channels": 180, "out_channels": 90, "is_first": false}, "shared_object_id": 13}]}}}2
�*root.siren_layers.2"_tf_keras_sequential*�{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 137]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.2.0_input"}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.0", "in_channels": 137, "out_channels": 90, "is_first": false}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.1", "in_channels": 90, "out_channels": 90, "is_first": false}}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.2", "in_channels": 90, "out_channels": 90, "is_first": false}}]}, "shared_object_id": 19, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 137]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 512, 512, 137]}, "float32", "siren_layers.2.0_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 512, 512, 137]}, "float32", "siren_layers.2.0_input"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 137]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "siren_layers.2.0_input"}, "shared_object_id": 15}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.0", "in_channels": 137, "out_channels": 90, "is_first": false}, "shared_object_id": 16}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.1", "in_channels": 90, "out_channels": 90, "is_first": false}, "shared_object_id": 17}, {"class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.2", "in_channels": 90, "out_channels": 90, "is_first": false}, "shared_object_id": 18}]}}}2
�3(root.siren_layers.0.layer_with_weights-0"_tf_keras_layer*�{"name": "siren_layers.0.0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.0", "in_channels": 47, "out_channels": 360, "is_first": true}, "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 47]}}2
�4(root.siren_layers.0.layer_with_weights-1"_tf_keras_layer*�{"name": "siren_layers.0.1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.1", "in_channels": 360, "out_channels": 360, "is_first": false}, "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 360]}}2
�5(root.siren_layers.0.layer_with_weights-2"_tf_keras_layer*�{"name": "siren_layers.0.2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.0.2", "in_channels": 360, "out_channels": 180, "is_first": false}, "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 360]}}2
�<(root.siren_layers.1.layer_with_weights-0"_tf_keras_layer*�{"name": "siren_layers.1.0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.0", "in_channels": 227, "out_channels": 180, "is_first": false}, "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 227]}}2
�=(root.siren_layers.1.layer_with_weights-1"_tf_keras_layer*�{"name": "siren_layers.1.1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.1", "in_channels": 180, "out_channels": 180, "is_first": false}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 180]}}2
�>(root.siren_layers.1.layer_with_weights-2"_tf_keras_layer*�{"name": "siren_layers.1.2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.1.2", "in_channels": 180, "out_channels": 90, "is_first": false}, "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 180]}}2
�E(root.siren_layers.2.layer_with_weights-0"_tf_keras_layer*�{"name": "siren_layers.2.0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.0", "in_channels": 137, "out_channels": 90, "is_first": false}, "shared_object_id": 16, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 137]}}2
�F(root.siren_layers.2.layer_with_weights-1"_tf_keras_layer*�{"name": "siren_layers.2.1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.1", "in_channels": 90, "out_channels": 90, "is_first": false}, "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 90]}}2
�G(root.siren_layers.2.layer_with_weights-2"_tf_keras_layer*�{"name": "siren_layers.2.2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "SineLinearLayer", "config": {"name": "siren_layers.2.2", "in_channels": 90, "out_channels": 90, "is_first": false}, "shared_object_id": 18, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 90]}}2
�
[/root.siren_layers.0.layer_with_weights-0.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 360, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.02127659574468085, "maxval": 0.02127659574468085, "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 47}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 47]}}2
�
b/root.siren_layers.0.layer_with_weights-1.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 360, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.004303314829119351, "maxval": 0.004303314829119351, "seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 360}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 360]}}2
�
i/root.siren_layers.0.layer_with_weights-2.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.004303314829119351, "maxval": 0.004303314829119351, "seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 360}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128, 128, 360]}}2
�
y/root.siren_layers.1.layer_with_weights-0.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.005419278146034048, "maxval": 0.005419278146034048, "seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 227}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 227]}}2
�
�/root.siren_layers.1.layer_with_weights-1.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 180, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.006085806194501845, "maxval": 0.006085806194501845, "seed": null}, "shared_object_id": 36}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 37}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 180}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 180]}}2
�
�/root.siren_layers.1.layer_with_weights-2.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.006085806194501845, "maxval": 0.006085806194501845, "seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 180}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 256, 256, 180]}}2
�
�/root.siren_layers.2.layer_with_weights-0.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.006975801064151558, "maxval": 0.006975801064151558, "seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 137}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 137]}}2
�
�/root.siren_layers.2.layer_with_weights-1.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.008606629658238702, "maxval": 0.008606629658238702, "seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 90}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 90]}}2
�
�/root.siren_layers.2.layer_with_weights-2.linear"_tf_keras_layer*�
{"name": "linear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv2D", "config": {"name": "linear", "trainable": true, "dtype": "float32", "filters": 90, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.008606629658238702, "maxval": 0.008606629658238702, "seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 90}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512, 512, 90]}}2