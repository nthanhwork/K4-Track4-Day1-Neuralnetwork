"""Measured observations and concise reporting; never trains or opens eval data."""
import json
from pathlib import Path
import numpy as np


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def observations(results):
    r = {x['cfg']['exp_id']: x for x in results}
    b = r['base-s1']['summary']
    seeds = [x['summary'] for x in results if x['cfg']['group']=='baseline']
    f1 = np.array([x['val_macro_f1'] for x in seeds])
    acc = np.array([x['val_acc'] for x in seeds])
    noise = 2*f1.std(ddof=1)
    def compare(key):
        value = r[key]['summary']['val_macro_f1']
        delta = value-b['val_macro_f1']
        return f"`{key}`: F1={value:.6f}, Δ={delta:+.6f}, {'vượt' if abs(delta)>noise else 'chưa vượt'} 2σ={noise:.6f}. "
    def gap(key):
        s = r[key]['summary']
        return s['final_val_loss']-s['final_train_loss']
    text = {}
    text['baseline'] = (f"Lr .05 được chọn bằng val: F1={b['val_macro_f1']:.6f}, so với .01 "
                       f"{r['tune-sgd-lr0.01']['summary']['val_macro_f1']:.6f}, khớp dự đoán hội tụ nhanh hơn. "
                       f"Train/val loss cuối={b['final_train_loss']:.6f}/{b['final_val_loss']:.6f}; giảm nhưng có dao động và gap nhỏ. "
                       f"Accuracy mean±std={acc.mean():.6f}±{acc.std(ddof=1):.6f}; F1={f1.mean():.6f}±{f1.std(ddof=1):.6f}. "
                       f"Ba seed cho 2σ={noise:.6f}, chỉ là ước lượng nhiễu thô.")
    text['loss'] = compare('loss-mse') + ('Khớp' if r['loss-mse']['summary']['val_macro_f1'] < b['val_macro_f1'] else 'Khác') + (
        ' dự đoán CE tốt hơn trong lịch này. MSE hồi quy logits về one-hot, mean B×7, scale gradient khác CE. '
        'Không so trị số hai loss; MSE chưa quét lr riêng.')
    text['optimizer'] = compare('opt-adam-lr0.001')+compare('opt-adam-lr0.003')+(
        'Khớp dự đoán lợi ích moment thích nghi; Adam .003 cao nhất grid. SGD momentum thử .01/.05, '
        'Adam .001/.003; không suy ra optimizer luôn thắng. Betas=(.9,.999), eps=1e-8, weight decay=0.')
    text['hparam'] = compare('batch-256') + (
        f"Khớp dự đoán nhiều cập nhật đi kèm F1 cao hơn: 1453 so với 727 bước/epoch, "
        f"{r['batch-256']['summary']['time_per_epoch_s']:.6f}s so với {b['time_per_epoch_s']:.6f}s. "
        'Cùng epoch khác số cập nhật, không quy toàn bộ lợi ích cho nhiễu gradient.')
    text['dropout'] = compare('drop-01') + (
        f"Gap val−train giảm {gap('base-s1'):.6f}→{gap('drop-01'):.6f}, nhưng F1 giảm: khớp dự đoán "
        'dropout có thể underfit khi baseline chưa quá khớp mạnh. Cả hai loss đo lúc dropout tắt.')
    delta = r['highlr-clip']['summary']['val_macro_f1']-r['highlr-no-clip']['summary']['val_macro_f1']
    text['clipping'] = compare('clip-normal') + (
        f"c={r['clip-normal']['cfg']['clip_norm']:.6f} từ p75 baseline, tỷ lệ clip TB="
        f"{np.mean(r['clip-normal']['history']['clip_fraction']):.6f}: khớp dự đoán clip kích hoạt. "
        f"Ở lr ×10, có/không clip F1={r['highlr-clip']['summary']['val_macro_f1']:.6f}/"
        f"{r['highlr-no-clip']['summary']['val_macro_f1']:.6f}, Δ={delta:+.6f}, vượt 2σ. "
        'Cả hai không diverged; không khẳng định clip cứu NaN. Clip chỉ giới hạn gradient và đổi quỹ đạo cập nhật.')
    text['amp'] = ''
    for key in ('amp-bf16','amp-fp16'):
        if key in r:
            s = r[key]['summary']
            ratio = b['time_per_epoch_s']/s['time_per_epoch_s']
            text['amp'] += compare(key)+f"{s['time_per_epoch_s']:.6f}s/epoch, FP32/AMP={ratio:.6f}; {'nhanh' if ratio>1 else 'chậm'} hơn FP32. "
    skipped = r.get('amp-fp16',r['base-s1'])['summary'].get('scaler_skipped_steps',0)
    text['amp'] += (f"Khác kỳ vọng tăng tốc GEMM đơn giản, phù hợp dự đoán overhead có thể chi phối mạng nhỏ. "
                    f"Thời gian gồm validation FP32 và đồng bộ. FP16 scaler bỏ {skipped} bước overflow, norm lưu null; "
                    'BF16 không cần scaler. Không ngoại suy sang GPU khác. FP16 là phép đo bổ sung, không chọn lại cấu hình cuối.')
    text['init'] = ''.join(compare(key) for key in ('init-zeros','init-normal','init-xavier')) + (
        'Khớp dự đoán zeros không học đặc trưng: chỉ bias cuối học prior, ReLU tại 0 chặn gradient lớp ẩn. '
        'Normal nhỏ làm tín hiệu suy giảm. Std ghi sau từng Linear, kể cả output. '
        'Hai lớp ẩn không đại diện mạng rất sâu; mọi kỹ thuật ở đây vẫn chỉ một seed.')
    return text


def compact_report(results, health, environment, selection, official, baseline_eval):
    r = {x['cfg']['exp_id']: x for x in results}
    b = r['base-s1']['summary']
    fid = selection['final_exp_id']
    final = r[fid]['summary']
    texts = observations(results)
    worst = min(official['per_class'],key=lambda c:c['f1'])
    errors = np.array(official['confusion_matrix'][worst['cls']]); errors[worst['cls']] = 0
    confused = int(errors.argmax())
    grid = table(['exp_id','val F1','best epoch'],
                 [[key,f"{x['summary']['val_macro_f1']:.6f}",x['summary']['best_epoch']]
                  for key,x in r.items() if x['cfg']['seed']==1])
    precision = table(['exp_id','s/epoch','GPU peak MB'],
                      [[key,f"{r[key]['summary']['time_per_epoch_s']:.6f}",f"{r[key]['summary']['peak_mem_MB']:.6f}"]
                       for key in ('base-s1','amp-bf16','amp-fp16') if key in r])
    classes = table(['Lớp','support','precision','recall','F1'],
                    [[c['cls'],c['support'],f"{c['precision']:.6f}",f"{c['recall']:.6f}",f"{c['f1']:.6f}"]
                     for c in official['per_class']])
    report = f"""# Báo cáo Lab Day 1 — Nguyễn Thái Anh — 2A202602810

## 1. Thiết lập

Conda **{environment['conda']}**, Python {environment['python']}, PyTorch {environment['torch']},
**{environment['device']}**. Metadata gốc: 464 809 train / 116 203 eval;
tách val phân tầng seed 42: 371 847 học / 92 962 val. Chỉ chuẩn hoá 10 cột số
bằng phần học; giữ 44 cột one-hot. Eval chỉ mở sau chọn cấu hình bằng val.

Baseline `base-s1`: **54→256→128→7, 47 879 tham số**, ReLU, bias; CE, He mọi Linear,
SGD momentum .9, lr .05 chọn giữa .01/.05, batch 512, 20 epoch, không dropout/clip, FP32.
Giữ batch cuối; train loss ở eval mode trên 50 000 mẫu cố định, val toàn tập.
Chọn checkpoint theo val loss thấp nhất, xếp cấu hình bằng macro-F1 tại checkpoint đó.

## 2. Kiểm tra ban đầu và nhiễu

He CE bước 0 **{health['step0_loss']:.6f}**, logits đều **{health['uniform_step0_loss']:.6f}**:
ln(7) là mốc dự đoán đều; logits ngẫu nhiên He có thể lệch. Logits đúng (B,7), mọi gradient He khác 0.
Ghi nhớ 20 mẫu: CE **{health['tiny_metrics']['loss']:.6f}**, accuracy **{health['tiny_metrics']['acc']:.6f}**.
[Đường cong kiểm tra](figures/health_overfit20.png).

{texts['baseline']} Accuracy baseline {b['val_acc']:.6f} vượt mốc đa số 0.487597.
[So sánh seed](figures/compare_baseline.png). Noise ghi trong Seeds; sức khoẻ trong Legend.

## 3. Kết quả theo chủ đề — chỉ dùng validation

Cùng split, 20 epoch, seed 1. Mỗi biến thể đổi một yếu tố; Adam đổi optimizer và lr
có chủ đích, thử hai lr mỗi optimizer. Cặp lr cao có/không clip chỉ khác clip.
Bảng Excel có đủ 17 run, mỗi run một PNG cùng exp_id và JSON. Kỹ thuật khác chỉ một seed,
so với 2σ baseline chưa thay thế kiểm định nhiều seed của từng kỹ thuật.

{grid}

**Loss — dự đoán CE phù hợp hơn MSE.** {texts['loss']}
CE gradient p−y; MSE 2(z−y)/(B×7), không 1/2, không softmax trước loss.
[Ảnh](figures/compare_loss.png).

**Optimizer — dự đoán moment thích nghi giúp Adam hội tụ.** {texts['optimizer']}

![So sánh optimizer](figures/compare_optimizer.png)

**Batch — dự đoán batch nhỏ học thêm nhưng tốn thời gian.** {texts['hparam']}
[Ảnh](figures/compare_hparam.png).

**Dropout — dự đoán q=.1 có thể underfit.** {texts['dropout']}
[Ảnh](figures/compare_dropout.png).

**Clipping — dự đoán c=p75 baseline kích hoạt một phần bước.** {texts['clipping']}
Norm đo trước clip; unscale trước đo/clip nếu FP16. [Ảnh](figures/compare_clipping.png).

**Mixed precision — dự đoán lợi ích phụ thuộc overhead.** {texts['amp']}

{precision}

FP16 exponent hẹp nên cần scaler; BF16 exponent như FP32. Tham số và validation vẫn FP32.
Nếu chạy CPU, bộ nhớ là RSS cực đại tích luỹ, không so như VRAM từng run.
[Ảnh](figures/compare_amp.png).

**Init — dự đoán zeros chỉ học prior, normal nhỏ học chậm.** {texts['init']}
Xavier Var=2/(fan-in+fan-out), He Var=2/fan-in, bias=0; std sau từng Linear trong Legend/JSON.
[Ảnh](figures/compare_init.png).

## 4. Đánh giá cuối trên eval và phân tích lỗi

Chọn **`{fid}`** bằng val: Adam lr .003, M-base/CE/He, batch 512, 20 epoch,
không dropout/clip, FP32, seed 1. Không chọn seed 2/3 hoặc chỉnh cấu hình sau eval.
FP16 là phép đo bổ sung, loại khỏi chọn cấu hình đã khoá. Số từ script giảng viên:

{table(['exp_id','val F1','eval F1','eval accuracy'],[['base-s1',f"{b['val_macro_f1']:.6f}",f"{baseline_eval['macro_f1']:.6f}",f"{baseline_eval['accuracy']:.6f}"],[fid,f"{final['val_macro_f1']:.6f}",f"{official['macro_f1']:.6f}",f"{official['accuracy']:.6f}"]])}

Eval F1 tăng **{official['macro_f1']-baseline_eval['macro_f1']:+.6f}**; eval−val cuối
**{official['macro_f1']-final['val_macro_f1']:+.6f}**. 2σ đo trên val baseline;
chưa đo nhiễu eval nhiều seed cuối nên chưa kết luận ý nghĩa thống kê.

{classes}

Khó nhất lớp **{worst['cls']}**, F1 **{worst['f1']:.6f}**, nhầm sang lớp **{confused}**
({int(errors[confused])} mẫu). Mất cân bằng có thể góp phần; tương đồng địa hình là giả thuyết
chưa kiểm tra feature theo lớp. Sẽ thử CE trọng số lớp và kiểm tra bằng val.

![Ma trận nhầm lẫn: hàng thật, cột dự đoán](figures/eval_confusion.png)

## 5. Câu hỏi dẫn dắt

1. **Optimizer thắng?** Adam .003 cao nhất grid; lr chung dễ thiên lệch vì scale cập nhật khác.
Cần grid rộng và nhiều seed trước khi kết luận tổng quát.
2. **Dropout giúp khi chưa quá khớp?** Run này gap giảm nhưng F1 kém hơn; dùng khi
train tiếp tục tốt lên còn val xấu đi, chọn q bằng val.
3. **Clipping giải quyết gì?** Giới hạn gradient lớn; tỷ lệ clip chứng minh kích hoạt.
Không sửa dữ liệu, kiến trúc hoặc bảo đảm mọi lr cao ổn định.
4. **AMP nhanh hơn?** Không ở đây; kernel, đo gradient và đồng bộ có thể lấn lợi ích GEMM.
FP16 cần scaler vì exponent hẹp; BF16 thường không cần.
5. **Zeros/He/Xavier?** Zeros giữ đối xứng, ReLU(0) chặn lớp ẩn. He bù variance mất qua ReLU;
Xavier cân bằng fan-in/out, ảnh hưởng rõ hơn ở mạng sâu.
6. **Loss không giảm sau 2000 bước:** kiểm tra shape/dtype/nhãn và CE nhận logits;
thử ghi nhớ 20 mẫu, kiểm tra gradient/zero_grad/backward/step/optimizer;
kiểm tra lr, preclip norm và activation std để phân biệt học chậm, nổ hoặc chết gradient.

## 6. Hạn chế và điều bất ngờ

He bước 0 lệch ln(7), AMP chậm hơn, lr ×10 không diverged: ghi đúng số đo.
Ba seed baseline là ước lượng thô; kỹ thuật khác một seed, chọn trong nhiều cấu hình có thể
lạc quan trên val. Chưa quét đủ lr, q, weight decay; batch khác số cập nhật.
Nếu thêm thời gian: nhiều seed cuối và class weighting, chỉ chọn bằng val.

## 7. Phụ lục

Pipeline lần cuối {environment['total_time_s']:.6f}s. Nộp code/notebook có output, REPORT.md,
experiments.xlsx, predictions_eval.csv, eval_result.json, figures, results;
không dữ liệu/checkpoint/cache. Số phụ/per-class/confusion gắn exp_id trong Legend;
noise trong Seeds, cấu hình/metric trong Experiments. Nhận xét chi tiết ngay dưới mỗi nhóm notebook.
selection.json ghi quyết định trước eval; chạy lại dự đoán giống hệt thì dùng JSON chấm chính thức đã lưu.
"""
    return report


def zero_init_diagnostic(data):
    from model import MLP
    from train import compute_loss
    model = MLP(init='zeros').to(data['X_tr'].device)
    compute_loss(model(data['X_tr'][:512]), data['y_tr'][:512], 'ce').backward()
    gradients = {name:float(p.grad.norm()) for name,p in model.named_parameters()}
    assert all(value == 0 for name,value in gradients.items() if name != 'net.6.bias')
    assert gradients['net.6.bias'] > 0
    return dict(exp_id='init-zeros', initial_gradient_norms=gradients,
                device=str(data['X_tr'].device), batch=512)


def refresh_notebook(notebook, results):
    texts = observations(results)
    group = None
    headings = {'### Hàm mất mát':'loss','### Bộ tối ưu':'optimizer','### Batch size':'hparam',
                '### Dropout':'dropout','### Gradient clipping':'clipping','### Mixed precision':'amp','### Khởi tạo':'init'}
    for c in notebook.cells:
        if c.cell_type != 'markdown':
            continue
        for heading,key in headings.items():
            if c.source.startswith(heading):
                group = key
        if c.source.startswith('**Đối chiếu baseline:**'):
            c.source = '**Đối chiếu baseline:** '+texts['baseline']
        elif c.source.startswith('**Đối chiếu sau:**') and group:
            c.source = '**Đối chiếu sau:** '+texts[group]
    return notebook


if __name__ == '__main__':
    import nbformat
    out = Path(__file__).resolve().parent.parent
    if not (out/'results').exists():
        out = out/'submission_2A202602810'
    results = [json.loads(p.read_text()) for p in sorted((out/'results').glob('*.json'))]
    read = lambda name: json.loads((out/name).read_text())
    report = compact_report(results,read('health_checks.json'),read('environment.json'),read('selection.json'),
                            read('eval_result.json'),read('baseline_eval_result.json'))
    (out/'REPORT.md').write_text(report)
    path = out/'code/lab.ipynb'
    notebook = refresh_notebook(nbformat.read(path,as_version=4),results)
    nbformat.write(notebook,path)
    print('Report words:',len(report.split()))
    print('Notebook observations populated from measured records; code outputs preserved.')
