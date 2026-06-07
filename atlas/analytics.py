# atlas/analytics.py
import os
import json
import time
from datetime import datetime
import numpy as np

# Safe imports for matplotlib to prevent crashes in headless environments
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MetricsRecorder:
    """
    Records and tracks training metrics per interaction, and generates
    detailed mathematical/visual reports at the end of a training session.
    """
    def __init__(self, config, metrics_history, start_time):
        self.config = config or {}
        self.metrics_history = metrics_history
        self.start_time = start_time
        
        reporting_config = self.config.get('reporting', {})
        self.enabled = reporting_config.get('enabled', True)
        self.output_dir = reporting_config.get('output_dir', '.reports/statistics')
        self.generate_graphs_enabled = reporting_config.get('generate_graphs', True)
        self.max_history_points = reporting_config.get('max_history_points', 1000)

    def add_point(self, loss, learning_rate, vocab_size):
        """
        Records a single training interaction step.
        """
        if not self.enabled:
            return

        # Determine next monotonic step number
        if self.metrics_history:
            step = self.metrics_history[-1]['step'] + 1
        else:
            step = 1

        # Cap memory history size
        if len(self.metrics_history) >= self.max_history_points:
            self.metrics_history.pop(0)

        self.metrics_history.append({
            'step': step,
            'loss': float(loss) if loss is not None else None,
            'learning_rate': float(learning_rate),
            'vocab_size': int(vocab_size),
            'timestamp': datetime.now().isoformat()
        })

    def generate_report(self, brain):
        """
        Triggers folder and file creation for raw metrics and report.
        """
        if not self.enabled:
            return

        if not self.metrics_history:
            print("No training metrics to report. Skipping report generation.")
            return

        # Create base report output dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Create timestamped folder (dd-mm-yyyy-hh-mm)
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
        report_dir = os.path.join(self.output_dir, timestamp)
        os.makedirs(report_dir, exist_ok=True)

        # 1. Save raw metrics as metrics.json
        metrics_json_path = os.path.join(report_dir, "metrics.json")
        try:
            with open(metrics_json_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"[✓] Saved raw metrics to {metrics_json_path}")
        except Exception as e:
            print(f"WARNING: Could not save raw metrics: {e}")

        # 2. Save PDF or fall back to Text report
        pdf_path = os.path.join(report_dir, "analytics.pdf")
        if HAS_MATPLOTLIB and self.generate_graphs_enabled:
            try:
                generate_pdf_report(
                    metrics_history=self.metrics_history,
                    config=self.config,
                    start_time=self.start_time,
                    end_time=datetime.now(),
                    output_path=pdf_path,
                    brain=brain
                )
                print(f"[✓] Generated PDF analytics report at {pdf_path}")
            except Exception as e:
                print(f"WARNING: Failed to generate PDF report using matplotlib: {e}")
                self._generate_text_report(report_dir, brain)
        else:
            if not HAS_MATPLOTLIB and self.generate_graphs_enabled:
                print("WARNING: matplotlib is not installed. Skipping PDF generation.")
            self._generate_text_report(report_dir, brain)

    def _generate_text_report(self, report_dir, brain):
        """
        Fallback text-based summary report when matplotlib is not available.
        """
        text_report_path = os.path.join(report_dir, "analytics_summary.txt")
        try:
            with open(text_report_path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("           ATLAS MODEL TRAINING ANALYTICS REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
                f.write(f"Start Time:   {self.start_time.strftime('%d-%m-%Y %H:%M:%S')}\n")
                f.write(f"End Time:     {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
                f.write(f"Duration:     {str(datetime.now() - self.start_time).split('.')[0]}\n")
                f.write(f"Total Steps:  {len(self.metrics_history)}\n\n")

                f.write("Model Architecture Configuration:\n")
                model_config = self.config.get('model', {})
                for k, v in model_config.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

                losses = [m['loss'] for m in self.metrics_history if m['loss'] is not None]
                if losses:
                    f.write("Training Metrics Statistics:\n")
                    f.write(f"  Initial Loss:          {losses[0]:.4f}\n")
                    f.write(f"  Final Loss:            {losses[-1]:.4f}\n")
                    f.write(f"  Minimum Loss:          {min(losses):.4f}\n")
                    f.write(f"  Maximum Loss:          {max(losses):.4f}\n")
                    f.write(f"  Mean Loss:             {np.mean(losses):.4f}\n")
                    f.write(f"  Std Dev Loss:          {np.std(losses):.4f}\n")
                f.write(f"  Final Learning Rate:   {self.metrics_history[-1]['learning_rate']:.6f}\n")
                f.write(f"  Final Vocabulary Size: {self.metrics_history[-1]['vocab_size']}\n\n")

                # Embedding statistics
                if hasattr(brain.transformer, 'token_embedding'):
                    token_emb = brain.transformer.token_embedding
                    norms = np.linalg.norm(token_emb, axis=1)
                    f.write("Token Embedding Statistics:\n")
                    f.write(f"  Embedding Shape:       {token_emb.shape}\n")
                    f.write(f"  Mean L2 Norm:          {np.mean(norms):.4f}\n")
                    f.write(f"  Std Dev L2 Norm:       {np.std(norms):.4f}\n")
                    f.write(f"  Min / Max L2 Norm:     {np.min(norms):.4f} / {np.max(norms):.4f}\n")
            print(f"[✓] Generated text analytics summary at {text_report_path}")
        except Exception as e:
            print(f"WARNING: Could not generate text analytics summary: {e}")


def generate_pdf_report(metrics_history, config, start_time, end_time, output_path, brain):
    """
    Generates a beautifully structured multi-page PDF using Matplotlib's backend_pdf.
    """
    # Use clean default/classic style or grid defaults if seaborn is missing
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    with PdfPages(output_path) as pdf:
        # ----------------------------------------------------
        # Page 1: Title & Executive Summary
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#f8fafc')  # Soft slate/gray background
        
        # Header banner
        ax_banner = fig.add_axes([0, 0.85, 1, 0.15])
        ax_banner.axis('off')
        ax_banner.fill_between([0, 1], 0, 1, color='#0f172a')  # Deep dark slate blue
        ax_banner.text(0.5, 0.5, "ATLAS TRAINING ANALYTICS REPORT", color='white', 
                       fontsize=22, fontweight='bold', ha='center', va='center')
        
        # Summary Content Area
        ax_content = fig.add_axes([0.1, 0.1, 0.8, 0.7])
        ax_content.axis('off')
        
        y_pos = 0.95
        ax_content.text(0, y_pos, "Training Session Summary", fontsize=16, fontweight='bold', color='#1e293b')
        y_pos -= 0.04
        ax_content.plot([0, 1], [y_pos, y_pos], color='#cbd5e1', lw=1.5)
        y_pos -= 0.05
        
        duration = end_time - start_time
        total_steps = len(metrics_history)
        losses = [m['loss'] for m in metrics_history if m['loss'] is not None]
        initial_loss = losses[0] if losses else 0.0
        final_loss = losses[-1] if losses else 0.0
        min_loss = min(losses) if losses else 0.0
        final_lr = metrics_history[-1]['learning_rate'] if metrics_history else 0.0
        initial_vocab = metrics_history[0]['vocab_size'] if metrics_history else 0
        final_vocab = metrics_history[-1]['vocab_size'] if metrics_history else 0
        
        summary_info = [
            ("Start Time", start_time.strftime("%d-%m-%Y %H:%M:%S")),
            ("End Time", end_time.strftime("%d-%m-%Y %H:%M:%S")),
            ("Training Duration", str(duration).split('.')[0]),
            ("Total Interactions (Steps)", f"{total_steps}"),
            ("Initial Loss", f"{initial_loss:.4f}"),
            ("Final Loss", f"{final_loss:.4f}"),
            ("Minimum Loss Achieved", f"{min_loss:.4f}"),
            ("Final Learning Rate", f"{final_lr:.6f}"),
            ("Initial Vocab Size", f"{initial_vocab}"),
            ("Final Vocab Size", f"{final_vocab}"),
        ]
        
        for label, val in summary_info:
            ax_content.text(0, y_pos, label, fontsize=11, fontweight='semibold', color='#475569')
            ax_content.text(0.45, y_pos, val, fontsize=11, color='#0f172a')
            y_pos -= 0.048
            
        y_pos -= 0.02
        ax_content.text(0, y_pos, "Model Architecture Configuration", fontsize=16, fontweight='bold', color='#1e293b')
        y_pos -= 0.04
        ax_content.plot([0, 1], [y_pos, y_pos], color='#cbd5e1', lw=1.5)
        y_pos -= 0.05
        
        model_config = config.get('model', {})
        config_info = [
            ("Embedding Dimension (embed_dim)", f"{model_config.get('embed_dim', 'N/A')}"),
            ("Attention Heads (num_heads)", f"{model_config.get('num_heads', 'N/A')}"),
            ("Feed-Forward Dimension (ff_dim)", f"{model_config.get('ff_dim', 'N/A')}"),
            ("Transformer Layers (num_layers)", f"{model_config.get('num_layers', 'N/A')}"),
            ("Max Sequence Length (max_seq_len)", f"{model_config.get('max_seq_len', 'N/A')}"),
            ("Dropout Rate (dropout_rate)", f"{model_config.get('dropout_rate', 'N/A')}")
        ]
        
        for label, val in config_info:
            ax_content.text(0, y_pos, label, fontsize=11, fontweight='semibold', color='#475569')
            ax_content.text(0.45, y_pos, val, fontsize=11, color='#0f172a')
            y_pos -= 0.048
            
        fig.text(0.5, 0.03, "Page 1 of 4", ha='center', fontsize=9, color='#64748b')
        pdf.savefig(fig)
        plt.close(fig)
        
        # ----------------------------------------------------
        # Page 2: Training Curves (Loss, LR, Vocab)
        # ----------------------------------------------------
        fig, axs = plt.subplots(3, 1, figsize=(8.5, 11))
        fig.patch.set_facecolor('#ffffff')
        plt.subplots_adjust(hspace=0.45, top=0.9, bottom=0.1, left=0.12, right=0.9)
        
        fig.suptitle("Training Metrics Progress", fontsize=18, fontweight='bold', color='#0f172a')
        
        steps = [m['step'] for m in metrics_history]
        losses_y = [m['loss'] for m in metrics_history]
        lrs_y = [m['learning_rate'] for m in metrics_history]
        vocabs_y = [m['vocab_size'] for m in metrics_history]
        
        # Plot 1: Loss over Time
        axs[0].plot(steps, losses_y, color='#f43f5e', lw=1.5, label='Loss per step')
        if len(losses_y) > 10:
            smooth_loss = np.convolve(losses_y, np.ones(10)/10, mode='valid')
            axs[0].plot(steps[9:], smooth_loss, color='#be123c', lw=2, linestyle='--', label='Smoothed Loss (MA-10)')
        axs[0].set_title("Loss Over Time", fontsize=12, fontweight='bold', color='#1e293b', loc='left')
        axs[0].set_ylabel("Loss", fontsize=10, color='#475569')
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend(fontsize=8, loc='upper right')
        
        # Plot 2: Learning Rate Schedule
        axs[1].plot(steps, lrs_y, color='#0ea5e9', lw=1.5)
        axs[1].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold', color='#1e293b', loc='left')
        axs[1].set_ylabel("Learning Rate", fontsize=10, color='#475569')
        axs[1].grid(True, linestyle=':', alpha=0.6)
        
        # Plot 3: Vocab Size Growth
        axs[2].plot(steps, vocabs_y, color='#10b981', lw=1.5)
        axs[2].set_title("Vocabulary Size Growth", fontsize=12, fontweight='bold', color='#1e293b', loc='left')
        axs[2].set_xlabel("Steps (Interactions)", fontsize=10, color='#475569')
        axs[2].set_ylabel("Vocab Size", fontsize=10, color='#475569')
        axs[2].grid(True, linestyle=':', alpha=0.6)
        
        fig.text(0.5, 0.03, "Page 2 of 4", ha='center', fontsize=9, color='#64748b')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # Page 3: Embedding Analysis
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#ffffff')
        
        fig.suptitle("Token Embedding Analysis", fontsize=18, fontweight='bold', color='#0f172a')
        
        token_emb = brain.transformer.token_embedding  # shape (vocab_size, embed_dim)
        emb_norms = np.linalg.norm(token_emb, axis=1)
        
        # Plot 1: Histogram of embedding norms
        ax_hist = fig.add_axes([0.12, 0.55, 0.76, 0.35])
        ax_hist.hist(emb_norms, bins=min(30, len(emb_norms)), color='#818cf8', edgecolor='#4f46e5', alpha=0.7)
        ax_hist.set_title("Distribution of Token Embedding L2 Norms", fontsize=12, fontweight='bold', color='#1e293b', loc='left')
        ax_hist.set_xlabel("L2 Norm Value", fontsize=10, color='#475569')
        ax_hist.set_ylabel("Frequency (Word Count)", fontsize=10, color='#475569')
        ax_hist.grid(True, linestyle=':', alpha=0.6)
        
        # Summary calculations
        mean_norm = np.mean(emb_norms)
        std_norm = np.std(emb_norms)
        min_norm = np.min(emb_norms)
        max_norm = np.max(emb_norms)
        p25, p50, p75 = np.percentile(emb_norms, [25, 50, 75])
        
        # Extract top & bottom words by norm
        words_norms = []
        for idx in range(len(emb_norms)):
            word = brain.idx_to_word.get(idx, f"<ID:{idx}>")
            words_norms.append((word, emb_norms[idx]))
            
        # Filter out special tokens for meaningful output
        filtered_words = [(w, n) for w, n in words_norms if w not in brain.SPECIAL_TOKENS]
        
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        top_words = filtered_words[:8]
        
        filtered_words.sort(key=lambda x: x[1])
        bottom_words = filtered_words[:8]
        
        # Text details
        ax_stats = fig.add_axes([0.12, 0.08, 0.76, 0.4])
        ax_stats.axis('off')
        
        y_pos = 0.95
        ax_stats.text(0, y_pos, "Embedding Norm Statistics", fontsize=13, fontweight='bold', color='#1e293b')
        y_pos -= 0.06
        
        stats_table = [
            ("Mean Embedding Norm", f"{mean_norm:.4f}"),
            ("Std Dev Embedding Norm", f"{std_norm:.4f}"),
            ("Min / Max Embedding Norm", f"{min_norm:.4f} / {max_norm:.4f}"),
            ("Percentiles (25% / 50% / 75%)", f"{p25:.4f} / {p50:.4f} / {p75:.4f}"),
        ]
        
        for label, val in stats_table:
            ax_stats.text(0, y_pos, label, fontsize=10, fontweight='semibold', color='#475569')
            ax_stats.text(0.5, y_pos, val, fontsize=10, color='#0f172a')
            y_pos -= 0.05
            
        y_pos -= 0.02
        ax_stats.text(0, y_pos, "Extreme Words by Embedding Norm", fontsize=13, fontweight='bold', color='#1e293b')
        y_pos -= 0.06
        
        ax_stats.text(0, y_pos, "Top 8 Highest Norms (Most Active)", fontsize=9.5, fontweight='bold', color='#4338ca')
        ax_stats.text(0.5, y_pos, "Top 8 Lowest Norms (Least Active)", fontsize=9.5, fontweight='bold', color='#4338ca')
        y_pos -= 0.05
        
        for k in range(8):
            top_str = f"{k+1}. {top_words[k][0]} ({top_words[k][1]:.3f})" if k < len(top_words) else ""
            bot_str = f"{k+1}. {bottom_words[k][0]} ({bottom_words[k][1]:.3f})" if k < len(bottom_words) else ""
            ax_stats.text(0, y_pos, top_str, fontsize=9, color='#334155')
            ax_stats.text(0.5, y_pos, bot_str, fontsize=9, color='#334155')
            y_pos -= 0.045
            
        fig.text(0.5, 0.03, "Page 3 of 4", ha='center', fontsize=9, color='#64748b')
        pdf.savefig(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # Page 4: Mathematical Analysis & Parameter Heatmap
        # ----------------------------------------------------
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#ffffff')
        
        fig.suptitle("Model Parameter & Mathematical Analysis", fontsize=18, fontweight='bold', color='#0f172a')
        
        # Heatmap of output layer projection matrix
        ax_heatmap = fig.add_axes([0.12, 0.55, 0.65, 0.35])
        
        out_layer = brain.transformer.output_layer  # shape (embed_dim, vocab_size)
        h_rows = min(40, out_layer.shape[0])
        h_cols = min(40, out_layer.shape[1])
        heatmap_data = out_layer[:h_rows, :h_cols]
        
        im = ax_heatmap.imshow(heatmap_data, cmap='coolwarm', aspect='auto')
        
        ax_cb = fig.add_axes([0.8, 0.55, 0.03, 0.35])
        fig.colorbar(im, cax=ax_cb, label='Weight Value')
        
        ax_heatmap.set_title(f"Output Projection Matrix Weights (Slice {h_rows}x{h_cols})", fontsize=12, fontweight='bold', color='#1e293b', loc='left')
        ax_heatmap.set_xlabel("Vocabulary Index", fontsize=9, color='#475569')
        ax_heatmap.set_ylabel("Embedding Index", fontsize=9, color='#475569')
        
        # Parameter analysis text
        ax_math = fig.add_axes([0.12, 0.06, 0.76, 0.42])
        ax_math.axis('off')
        
        y_pos = 0.98
        ax_math.text(0, y_pos, "Model Complexity and Parameter Weight Norms", fontsize=13, fontweight='bold', color='#1e293b')
        y_pos -= 0.05
        
        param_stats = []
        for name, param in brain.transformer.params.items():
            if name in ['token_embedding', 'output_layer', 'output_bias'] or 'attn_W_q' in name or 'ff_W1' in name:
                l2_norm = float(np.linalg.norm(param))
                p_mean = float(np.mean(param))
                p_std = float(np.std(param))
                p_name = name.replace('block_', 'Block ').replace('_attn_', ' Attention ').replace('_ff_', ' FeedForward ').replace('_', ' ')
                param_stats.append((p_name, f"{param.shape}", f"{l2_norm:.2f}", f"{p_mean:.4f} \u00b1 {p_std:.4f}"))
                
        # Draw parameter table
        ax_math.text(0, y_pos, "Parameter Group", fontsize=9, fontweight='bold', color='#475569')
        ax_math.text(0.4, y_pos, "Shape", fontsize=9, fontweight='bold', color='#475569')
        ax_math.text(0.65, y_pos, "L2 Norm", fontsize=9, fontweight='bold', color='#475569')
        ax_math.text(0.8, y_pos, "Mean \u00b1 Std", fontsize=9, fontweight='bold', color='#475569')
        y_pos -= 0.04
        ax_math.plot([0, 1], [y_pos + 0.015, y_pos + 0.015], color='#cbd5e1', lw=1)
        
        for p_name, p_shape, p_norm, p_dist in param_stats[:6]:
            ax_math.text(0, y_pos, p_name[:25], fontsize=8, color='#0f172a')
            ax_math.text(0.4, y_pos, p_shape, fontsize=8, color='#0f172a')
            ax_math.text(0.65, y_pos, p_norm, fontsize=8, color='#0f172a')
            ax_math.text(0.8, y_pos, p_dist, fontsize=8, color='#0f172a')
            y_pos -= 0.038
            
        y_pos -= 0.02
        ax_math.text(0, y_pos, "Logits Entropy Analysis (Model Confidence)", fontsize=13, fontweight='bold', color='#1e293b')
        y_pos -= 0.05
        
        # Calculate prediction entropy over sample input
        try:
            vocab_size = brain.vocab_size
            sample_size = min(100, vocab_size)
            dummy_vocab_emb = token_emb[:sample_size].astype(np.float32)
            dummy_logits = dummy_vocab_emb @ out_layer.astype(np.float32) + brain.transformer.output_bias.astype(np.float32)[:vocab_size]
            shift_logits = dummy_logits - np.max(dummy_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(shift_logits)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            entropy = -np.sum(probs * np.log(probs + 1e-9), axis=-1)
            mean_entropy = np.mean(entropy)
            max_possible_entropy = np.log(vocab_size)
            entropy_ratio = mean_entropy / max_possible_entropy if max_possible_entropy > 0 else 0
            
            entropy_msg = (
                f"Computed average softmax prediction entropy over first {sample_size} vocabulary tokens:\n"
                f"  - Average Entropy: {mean_entropy:.4f} nats\n"
                f"  - Maximum Possible Entropy (uniform distribution): {max_possible_entropy:.4f} nats\n"
                f"  - Confidence Index (1.0 - Entropy Ratio): {1.0 - entropy_ratio:.4f}\n"
                f"A higher Confidence Index indicates that the model has focused attention on specific associations\n"
                f"rather than maintaining a uniform prediction distribution."
            )
        except Exception as e:
            entropy_msg = f"Failed to compute prediction entropy analysis: {e}"
            
        ax_math.text(0, y_pos, entropy_msg, fontsize=8.5, color='#334155', linespacing=1.3)
        
        fig.text(0.5, 0.03, "Page 4 of 4", ha='center', fontsize=9, color='#64748b')
        pdf.savefig(fig)
        plt.close(fig)
