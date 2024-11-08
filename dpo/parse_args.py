def parse_args(parser):
    parser.add_argument(
        "--oracle_path",
        type=str,
        required=True,
        help="Path to the oracle model",
    )
    parser.add_argument(
        "--student_path",
        type=str,
        required=True,
        help="Path to the student model",
    )
    parser.add_argument(
        "--num_subtopic_per_topic",
        type=int,
        default=3,
        help="Number of subtopics per topic",
    )
    parser.add_argument(
        "--num_exercise_per_subtopic",
        type=int,
        default=3,
        help="Number of exercises per subtopic",
    )
    parser.add_argument(
        "--oracle_temperature",
        type=float,
        default=1.0,
        help="Temperature for oracle text generation",
    )
    parser.add_argument(
        "--oracle_max_length",
        type=int,
        default=2500,
        help="Maximum length of oracle generation",
    )
    parser.add_argument(
        "--student_temperature",
        type=float,
        default=0.6,
        help="Temperature for student text generation",
    )
    parser.add_argument(
        "--student_max_length",
        type=int,
        default=512,
        help="Maximum length of student generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="Number of training steps",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5.0e-6,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="Beta for DPO training",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        choices=range(1, 21),
    )
    parser.add_argument(
        "--use_existing_dataset",
        action='store_true'
    )
    parser.add_argument(
        "--do_quantization",
        action='store_true'
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Output path for the generated dataset",
    )


    args = parser.parse_args()

    return args
