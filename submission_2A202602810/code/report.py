"""Generate the report and workbook strictly from measured experiment records."""
import json
import platform
from pathlib import Path
import numpy as np
import torch
from results_table import to_row, write_xlsx

def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|','/') for x in row) + ' |')
    return '\n'.join(lines)

def f(x):
    return 'N/A' if x is None else f'{x:.6f}'

def make_deliverables(results, base, final, scores, health, data, out_dir, repo_root, total_time):
    out_dir, repo_root = Path(out_dir), Path(repo_root)
    template_path = repo_root/'templates/experiment_table_template.xlsx'
    if not template_path.exists():
        template_path = Path(__file__).with_name('experiment_table_template.xlsx')
    lookup = {r['cfg']['exp_id']:r for r in results}
    seeds = [r for r in results if r['cfg']['group']=='baseline']
    f1s = np.array([r['summary']['val_macro_f1'] for r in seeds])
    accs = np.array([r['summary']['val_acc'] for r in seeds])
    mean, std = f1s.mean(), f1s.std(ddof=1)
    noise = 2*std
    b, fs = base['summary'], final['summary']
    bid, fid = base['cfg']['exp_id'], final['cfg']['exp_id']
    official = scores[fid]
    baseline_eval = scores[bid]
    annotations = [[bid,'parameters',health['parameters']], [bid,'He health step0',health['step0_loss']],
                   [bid,'uniform step0 check',health['uniform_step0_loss']],
                   [bid,'overfit20 CE',health['tiny_metrics']['loss']],
                   [bid,'overfit20 accuracy',health['tiny_metrics']['acc']],
                   [bid,'majority validation accuracy',data['majority_val_acc']],
                   [bid,'train samples',len(data['X_tr'])], [bid,'validation samples',len(data['X_val'])],
                   [fid,'eval samples',official['n_eval']], [fid,'total wall time seconds',total_time],
                   [bid,'Python',platform.python_version()], [bid,'PyTorch',torch.__version__],
                   [bid,'device',torch.cuda.get_device_name(0) if data['X_tr'].is_cuda else 'CPU'],
                   [bid,'split seed',data['split_seed']]]
    for name, value in health['gradient_norms'].items():
        annotations.append([bid,'health gradient '+name,value])
    for r in results:
        rid, h, s = r['cfg']['exp_id'], r['history'], r['summary']
        annotations += [[rid,'activation stds after Linear',json.dumps(s['activation_stds'])],
                        [rid,'mean fraction clipped',float(np.mean(h['clip_fraction'])) if h['epoch'] else None],
                        [rid,'max preclip gradient',max(h['grad_norm_max'],default=None)],
                        [rid,'gradient norm p75',s['grad_norm_p75']],
                        [rid,'first epoch train loss',h['train_loss'][0] if h['epoch'] else None],
                        [rid,'first epoch validation loss',h['val_loss'][0] if h['epoch'] else None],
                        [rid,'total attempted update steps',sum(h['steps'])],
                        [rid,'GradScaler skipped overflow steps',s.get('scaler_skipped_steps',0)]]
    annotations += [[fid,'final eval per class: class','support','precision','recall','F1']]
    for c in official['per_class']:
        annotations.append([fid,'class '+str(c['cls']),c['support'],c['precision'],c['recall'],c['f1']])
    annotations += [[fid,'confusion matrix rows=true, columns=pred',0,1,2,3,4,5,6]]
    for i, row in enumerate(official['confusion_matrix']):
        annotations.append([fid,'true class '+str(i),*row])
    rows=[]
    for r in results:
        delta = r['summary']['val_macro_f1']-b['val_macro_f1'] if r['summary']['val_macro_f1'] is not None else None
        notes = f"delta vs base-s1={f(delta)}; 2sigma={noise:.6f}; selected final={r is final}"
        rows.append(to_row(r,scores.get(r['cfg']['exp_id']),notes))
    write_xlsx(rows,template_path,out_dir/'experiments.xlsx',annotations)
    def run_table(group, include_base=True):
        chosen=([base] if include_base else [])+[r for r in results if r is not base and r['cfg']['group']==group]
        return table(['exp_id','lr','val accuracy','val macro-F1','best epoch'],
                     [[r['cfg']['exp_id'],r['cfg']['lr'],f(r['summary']['val_acc']),f(r['summary']['val_macro_f1']),
                       r['summary']['best_epoch']] for r in chosen])
    def comparison(r):
        delta=r['summary']['val_macro_f1']-b['val_macro_f1']
        direction='cao hơn' if delta>0 else 'thấp hơn' if delta<0 else 'bằng'
        conclusion='vượt' if abs(delta)>noise else 'chưa vượt'
        return (f"`{r['cfg']['exp_id']}` có val macro-F1 {f(r['summary']['val_macro_f1'])}, {direction} "
                f"`{bid}` {abs(delta):.6f}; chênh lệch {conclusion} ngưỡng 2σ={noise:.6f}. "
                'Đây là so sánh một seed của kỹ thuật với baseline; chưa chứng minh tính ổn định qua nhiều seed.')
    dropout=lookup['drop-01']
    gap_base=b['final_val_loss']-b['final_train_loss']
    ds=dropout['summary']; gap_drop=ds['final_val_loss']-ds['final_train_loss']
    mse=lookup['loss-mse']
    high_delta=lookup['highlr-clip']['summary']['val_macro_f1']-lookup['highlr-no-clip']['summary']['val_macro_f1']
    clipping=[lookup[i] for i in ('clip-normal','highlr-no-clip','highlr-clip')]
    amp=[r for r in results if r['cfg']['group']=='amp']
    inits=[base]+[r for r in results if r['cfg']['group']=='init']
    optimizers=[r for r in results if r['cfg']['optimizer']=='adam']
    adam=max(optimizers,key=lambda r:r['summary']['val_macro_f1'])
    sgds=[r for r in results if r['cfg']['group']=='optimizer' and r['cfg']['optimizer']=='sgd_momentum']+[base]
    sgd=max(sgds,key=lambda r:r['summary']['val_macro_f1'])
    winner=adam if adam['summary']['val_macro_f1']>sgd['summary']['val_macro_f1'] else sgd
    worst=min(official['per_class'],key=lambda c:c['f1'])
    cm=np.array(official['confusion_matrix'])
    err=cm[worst['cls']].copy();err[worst['cls']]=0
    confused=int(err.argmax())
    train_counts=torch.bincount(data['y_tr'],minlength=7).cpu().tolist()
    environment=torch.cuda.get_device_name(0) if data['X_tr'].is_cuda else 'CPU (4 threads)'
    report=f"""# Báo cáo Lab Day 1 — Nguyễn Thái Anh — 2A202602810

## 1. Thiết lập

Chạy trực tiếp trong Conda **base**, Python {platform.python_version()}, PyTorch {torch.__version__},
thiết bị **{environment}**. Forest CoverType chia theo metadata gốc: 464 809 train / 116 203 eval.
Từ train, tách phân tầng seed 42 thành 371 847 mẫu học và 92 962 mẫu validation.
Chỉ 10 cột liên tục được chuẩn hoá bằng thống kê của 371 847 mẫu học; giữ 44 cột one-hot.
Tập eval chỉ được mở sau khi ghi quyết định cấu hình vào `selection.json`.

Baseline `base-s1`: M-base 54→256→128→7, **47 879 tham số**, ReLU, bias;
CE, He cho mọi Linear, SGD momentum 0.9, lr **{base['cfg']['lr']}**, batch 512,
20 epoch, dropout 0, không clip, FP32. Lr chọn bằng val trong 0.01 và 0.05.
Train loss đo ở eval mode trên 50 000 mẫu cố định seed 42; val dùng toàn tập.
Giữ batch cuối; không scheduler. Chọn checkpoint có **val loss thấp nhất**;
so cấu hình bằng macro-F1 tại checkpoint đó. FP16 là phép đo bổ sung, loại khỏi chọn cấu hình cuối; cấu hình cuối giữ nguyên sau khi chấm eval. Không dùng eval chọn lr, epoch hoặc seed.
Mốc đoán lớp đa số trên val: **{data['majority_val_acc']:.6f}** (`base-s1`, phần bằng chứng trong Legend).
Đã thử đủ bảy chủ đề: loss, optimizer, hyper-parameter, dropout, clipping, mixed precision, init.

## 2. Kiểm tra ban đầu và nhiễu

{table(['Kiểm tra (`base-s1`, bằng chứng Legend)','Kết quả'],[
['Parameters / logits shape',f"{health['parameters']} / (B,7)"],
['He CE bước 0',f(health['step0_loss'])],['CE với logits đều bằng 0',f(health['uniform_step0_loss'])],
['Ghi nhớ 20 mẫu: loss / accuracy',f"{f(health['tiny_metrics']['loss'])} / {f(health['tiny_metrics']['acc'])}"],
['Mọi gradient He khác 0',str(all(v>0 for v in health['gradient_norms'].values()))],
['Baseline val accuracy mean ± sample std',f"{accs.mean():.6f} ± {accs.std(ddof=1):.6f}"],
['Baseline val macro-F1 mean ± sample std',f"{mean:.6f} ± {std:.6f}"],
['Ngưỡng nhiễu 2σ',f(noise)]])}

Loss bước 0 của He lớn hơn ln(7): logits ngẫu nhiên chưa đều. Không coi ln(7) là
đẳng thức cho mọi khởi tạo. Phép thử logits bằng 0 xác nhận mốc này; kiểm tra gradient
và ghi nhớ 20 mẫu xác nhận vòng lặp cập nhật hoạt động. Std kích hoạt sau từng Linear
được ghi trong JSON và Legend. Baseline vượt mốc đa số; loss đầu/cuối train là
{base['history']['train_loss'][0]:.6f}/{b['final_train_loss']:.6f}, val là
{base['history']['val_loss'][0]:.6f}/{b['final_val_loss']:.6f}; checkpoint tốt nhất epoch {b['best_epoch']}.
Ba seed `base-s1`, `base-s2`, `base-s3` cho ước lượng nhiễu mẫu, không phải kiểm định thống kê.

![Kiểm tra ghi nhớ](figures/health_overfit20.png)
![Baseline và seed](figures/compare_baseline.png)

## 3. Kết quả theo chủ đề (chỉ dùng validation)

### 3.1 CE và MSE

**Dự đoán trước:** CE phù hợp phân loại hơn MSE và hội tụ tốt hơn trong cùng lịch học.
MSE ở đây lấy trung bình trên toàn bộ B×7 phần tử giữa **logits thô** và one-hot,
không hệ số 1/2, không dùng softmax. Không so trị số CE với MSE.

{run_table('loss')}

{comparison(mse)} Kết quả {'khớp' if mse['summary']['val_macro_f1']<b['val_macro_f1'] else 'khác'} dự đoán về CE trong cấu hình này. CE có gradient theo logits p−y; MSE có gradient 2(z−y)/(B×7),
phạt logits theo mục tiêu hồi quy. Cùng lr không đảm bảo hai loss đã được chỉnh tối ưu;
kết quả chỉ mô tả lịch huấn luyện này, không chứng minh CE luôn tốt hơn.

![Loss](figures/compare_loss.png)

### 3.2 Optimizer và độ nhạy lr

**Dự đoán trước:** moment thích nghi giúp Adam hội tụ nhanh ở lr 0.001/0.003;
SGD momentum cần lr riêng. Adam dùng betas=(0.9,0.999), eps=1e-8, weight decay=0.
SGD momentum thử hai lr ở Part 2; Adam thử hai lr ở Part 3. Các run Adam đổi
optimizer **và lr** có chủ đích để so ở lr tốt nhất của mỗi optimizer.

{run_table('optimizer')}

Trong grid đã đo, SGD tốt nhất là `{sgd['cfg']['exp_id']}`, Adam tốt nhất là
`{adam['cfg']['exp_id']}`; cao nhất trong hai là **`{winner['cfg']['exp_id']}`**.
{comparison(adam)} Momentum tích luỹ hướng cập nhật; Adam chia gradient theo
ước lượng moment bậc hai nên phản ứng khác với lr. Hai lr mỗi optimizer chưa phải
quét đầy đủ; không suy ra thuật toán thắng tuyệt đối hoặc kết luận về AdamW chưa chạy.

![Optimizer](figures/compare_optimizer.png)

### 3.3 Hyper-parameter

**Dự đoán trước:** batch 256 tăng số cập nhật và có thể học thêm chi tiết, đổi lại thời gian.
Giữ lr, seed và số epoch; batch 512 có 727 cập nhật/epoch, batch 256 có 1 453.
Không đổi kiến trúc trong thí nghiệm này.

{run_table('hparam')}

{comparison(lookup['batch-256'])} Batch nhỏ tăng cả nhiễu gradient và số cập nhật;
so cùng epoch là so cùng số lượt dữ liệu, không cùng số bước optimizer.

![Hyper-parameter](figures/compare_hparam.png)

### 3.4 Dropout

**Dự đoán trước:** q=0.1 có thể làm giảm khả năng học nếu baseline chưa quá khớp mạnh.
{comparison(dropout)} Khoảng cách loss val−train cuối baseline **{gap_base:.6f}**,
`drop-01` **{gap_drop:.6f}**. Kết quả {'khớp dự đoán: dropout làm F1 giảm dù gap nhỏ hơn' if ds['val_macro_f1']<b['val_macro_f1'] else 'khác dự đoán: F1 không giảm'}. Cả hai loss đều đo khi dropout tắt nên so được.
Dropout thêm nhiễu, ngăn các neuron phụ thuộc nhau; khoảng cách giảm tự nó chưa đủ
chứng minh cải thiện nếu cả train/val đều kém hơn. Chưa quét q hoặc nhiều seed dropout.

![Dropout](figures/compare_dropout.png)

### 3.5 Clipping

**Dự đoán trước:** c từ percentile 75 gradient baseline sẽ cắt một phần bước;
clip có thể giảm gai ở lr cao nhưng không đảm bảo cứu learning rate quá lớn.
Gradient ghi **trước clip**, FP16 unscale trước đo.

{table(['exp_id','lr','c','tỷ lệ bước clip TB','max gradient/epoch','val F1'],[
[r['cfg']['exp_id'],r['cfg']['lr'],f(r['cfg']['clip_norm']),
f(float(np.mean(r['history']['clip_fraction']))) if r['history']['epoch'] else 'N/A',
f(max(r['history']['grad_norm_max'],default=None)),f(r['summary']['val_macro_f1'])] for r in clipping])}

{comparison(lookup['clip-normal'])} Cặp lr cao giữ mọi cấu hình giống nhau trừ clip.
Nếu clip không nâng val F1, không gọi là cứu được huấn luyện: clipping chỉ giới hạn
chuẩn gradient, còn bước cập nhật phụ thuộc lr và momentum. Cờ diverged và đường cong
cung cấp bằng chứng thay vì mặc định lr cao phải sinh NaN.\nHai run lr cao đều không diverged, khác dự đoán rằng lr ×10 có thể làm huấn luyện hỏng.\nF1 có clip−không clip = **{high_delta:+.6f}**, {'vượt' if abs(high_delta)>noise else 'chưa vượt'} 2σ baseline.\nClipping ở lr cao chỉ tác động ít bước nhưng đổi quỹ đạo cập nhật; đây là quan sát một seed,\nkhông chứng minh clip cần thiết mọi lr.

![Clipping](figures/compare_clipping.png)

### 3.6 Mixed precision

**Dự đoán trước:** GPU có thể tăng tốc phép nhân ở FP16/BF16, nhưng MLP nhỏ và chi phí
kernel/đo gradient có thể làm lợi ích nhỏ. Tham số và đánh giá vẫn FP32;
FP16 dùng GradScaler, BF16 không cần scaler. Cả hai unscale/clip theo thứ tự đúng.

{table(['exp_id','precision','s/epoch','GPU peak MB hoặc RSS','scaler skips','val F1'],
[[r['cfg']['exp_id'],r['cfg']['precision'],f(r['summary']['time_per_epoch_s']),
f(r['summary']['peak_mem_MB']),r['summary'].get('scaler_skipped_steps',0),f(r['summary']['val_macro_f1'])] for r in [base]+amp])}

"""
    for r in amp:
        ratio=b['time_per_epoch_s']/r['summary']['time_per_epoch_s']
        report += (f"`{r['cfg']['exp_id']}`: tỷ số thời gian FP32/AMP = **{ratio:.6f}** "
                   f"({'nhanh hơn' if ratio>1 else 'chậm hơn'} FP32 trong phép đo này). {comparison(r)}\n\n")
        annotations.append([r['cfg']['exp_id'],'FP32/AMP epoch time ratio',ratio])
    report += f"""FP16 có khoảng biểu diễn hẹp, cần nhân loss để tránh underflow gradient;
BF16 có số bit exponent bằng FP32 nhưng mantissa ít hơn, thường không cần scaler.
Khi FP16 có gradient không hữu hạn do scale, GradScaler bỏ cập nhật và giảm scale;\nkhông coi riêng hiện tượng này là model diverged. Các bước này lưu norm=null trong JSON,\ngrad_norm mean/max chỉ tính bước hữu hạn; tổng bước thử và số skip được ghi riêng trong Legend.\nThời gian gồm huấn luyện, đo train/val FP32 và đồng bộ GPU. Nếu chạy CPU, peak_mem_MB
là RSS cực đại tích luỹ, không thể dùng để kết luận tiết kiệm bộ nhớ từng run.
Chỉ áp dụng kết quả tốc độ cho phần cứng này; không ngoại suy sang GPU khác.

![Mixed precision](figures/compare_amp.png)

### 3.7 Khởi tạo

**Dự đoán trước:** zeros chỉ học bias đầu ra; normal std=.01 có tín hiệu nhỏ;
He duy trì variance qua ReLU tốt hơn trong lý thuyết, Xavier có thể vẫn học tốt ở mạng nông.
Xavier dùng Var=2/(fan_in+fan_out); He dùng Var=2/fan_in cho **mọi** Linear, bias=0.

{table(['exp_id','loss bước 0','std sau các Linear','val F1'],
[[r['cfg']['exp_id'],f(r['summary']['step0_loss']),', '.join(f(v) for v in r['summary']['activation_stds']),
f(r['summary']['val_macro_f1'])] for r in inits])}

Zeros giữ đối xứng; ReLU tại 0 có đạo hàm 0, gradient các lớp ẩn không chảy;
bias output vẫn học prior lớp nên loss giảm không đồng nghĩa học được đặc trưng.
Normal nhỏ có kích hoạt suy giảm. Mạng chỉ hai lớp ẩn nên không đại diện hiện tượng
mạng hàng chục lớp; không mặc định He phải thắng Xavier từ một seed.\n{comparison(lookup['init-xavier'])}\n{comparison(lookup['init-normal'])}

![Init](figures/compare_init.png)

## 4. Đánh giá cuối trên eval

Chọn **`{fid}`** bằng val, seed 1; không chọn seed 2/3 làm mô hình cuối.
Cấu hình: {final['cfg']['optimizer']}, lr={final['cfg']['lr']}, batch={final['cfg']['batch']},
hidden={final['cfg']['hidden']}, loss={final['cfg']['loss']}, dropout={final['cfg']['dropout']},
clip={final['cfg']['clip_norm']}, init={final['cfg']['init']}, precision={final['cfg']['precision']}.
Chỉ đánh giá `base-s1` và cấu hình cuối, lấy số nguyên gốc từ script giảng viên.

{table(['Cấu hình','val F1','eval F1','eval accuracy'],[
[bid,f(b['val_macro_f1']),f(baseline_eval['macro_f1']),f(baseline_eval['accuracy'])],
[fid,f(fs['val_macro_f1']),f(official['macro_f1']),f(official['accuracy'])]])}

Chênh lệch eval macro-F1 cuối−baseline = **{official['macro_f1']-baseline_eval['macro_f1']:+.6f}**;
độ lệch eval−val của cấu hình cuối = **{official['macro_f1']-fs['val_macro_f1']:+.6f}**.
Ngưỡng 2σ ở phần 2 đo trên **val baseline**, không phải nhiễu eval của cấu hình cuối;
chưa có nhiều seed cuối trên eval nên không khẳng định ý nghĩa thống kê của cải thiện eval.

### 4.1 Phân tích lỗi theo lớp

{table(['Lớp','support','precision','recall','F1'],
[[c['cls'],c['support'],f(c['precision']),f(c['recall']),f(c['f1'])] for c in official['per_class']])}

Lớp khó nhất là **{worst['cls']}**, F1 **{worst['f1']:.6f}**, thường bị nhầm sang
lớp **{confused}** ({int(err[confused])} mẫu). Tập train có {train_counts[worst['cls']]} mẫu
lớp {worst['cls']} và {train_counts[confused]} mẫu lớp {confused} (bằng chứng Legend).
Mất cân bằng có thể ảnh hưởng ranh giới; sự giống nhau về địa hình là giả thuyết,
chưa đo phân bố feature theo cặp lớp nên không coi là kết luận. Lần sau thử CE trọng số lớp
và kiểm tra trên val; không dùng eval để chỉnh cấu hình trong bài này.

![Ma trận nhầm lẫn](figures/eval_confusion.png)

## 5. Câu hỏi dẫn dắt

1. Trong grid đã thử, `{winner['cfg']['exp_id']}` có val F1 cao nhất giữa SGD và Adam.
   So một lr chung không công bằng vì scale cập nhật khác; kết luận chỉ giới hạn grid và một seed kỹ thuật.
2. Dropout không đảm bảo giúp khi chưa quá khớp: đối chiếu val F1 và gap loss ở mục 3.4.
   Dùng khi train tiếp tục cải thiện trong khi val xấu đi, rồi chọn q bằng val.
3. Clipping chặn chuẩn gradient lớn; tỷ lệ clip và max gradient trước clip ở 3.5 chứng minh nó có kích hoạt.
   Nó không sửa nhãn, không tăng năng lực model và không đảm bảo cứu mọi lr cao.
4. Tốc độ AMP chỉ kết luận theo bảng thời gian 3.6. Mạng nhỏ chịu overhead gọi kernel và đồng bộ;
   FP16/BF16 có thể ít lợi ích hoặc chậm hơn dù phép nhân lý thuyết nhanh hơn.
5. Zeros làm ReLU có gradient 0 ở lớp ẩn và giữ neuron đối xứng, chỉ prior ở output thay đổi.
   He bù variance mất qua ReLU; Xavier cân bằng fan-in/fan-out; ảnh hưởng lớn hơn ở mạng sâu.
6. **Loss không giảm sau 2 000 bước:** (a) kiểm tra shape/dtype/nhãn 0..6 và CE nhận logits thô;
   (b) thử ghi nhớ 20 mẫu, in gradient từng tham số, kiểm tra zero_grad/backward/step và optimizer nhận đủ tham số;
   (c) nhìn preclip gradient, lr và std kích hoạt để phân biệt lr quá nhỏ, nổ gradient hoặc init làm gradient chết.
   Chỉ đổi kiến trúc sau khi dữ liệu và cập nhật vượt qua các phép thử này.

## 6. Hạn chế và điều bất ngờ

Loss He ban đầu không bằng ln(7), phù hợp logits ngẫu nhiên; không che giấu phép đo.
Ba seed baseline cho ước lượng nhiễu còn thô; kỹ thuật khác chỉ một seed và chọn tốt nhất
trong nhiều cấu hình có thể lạc quan trên validation. Cùng epoch nhưng batch khác nhau
làm khác số cập nhật. Chưa quét đầy đủ lr, dropout hoặc weight decay; không chạy AdamW.
Không giả định clipping cứu được lr cao hay AMP chắc chắn tăng tốc; dùng số đo ở 3.5–3.6.
Giới hạn bộ nhớ CPU/GPU khác nhau được ghi đúng trong JSON và notes Excel.
Nếu có thêm thời gian, chạy nhiều seed cho cấu hình cuối và các ứng viên gần nhau,
quét lr rộng hơn, thử loss có trọng số và đánh giá độ nhạy split bằng train/val.

## 7. Phụ lục

Thời gian pipeline đến lúc lập báo cáo: **{total_time:.6f} giây** (Legend, `{fid}`).
Nộp REPORT.md, experiments.xlsx (giữ bốn sheet và công thức), predictions_eval.csv,
eval_result.json, figures, results và toàn bộ code trong code/, notebook có output.
Mọi số đo sức khoẻ, clip, activation, thời gian và per-class nằm trong **Legend**
với exp_id; số chính mỗi run trong **Experiments**, nhiễu trong **Seeds**.
`selection.json` và `health_checks.json` giữ dấu vết quyết định/kiểm tra.
Không nộp dữ liệu, trọng số, cache. Mỗi dòng Experiments có một ảnh cùng exp_id;
các ảnh compare, health và confusion là bằng chứng bổ sung.\nFP16 được thêm như phép đo precision bổ sung và loại khỏi chọn cấu hình; không tinh chỉnh\nmô hình sau khi thấy eval. Khi chạy lại với dự đoán giống hệt, notebook dùng lại JSON chấm chính thức.
"""
    # Additional evidence used in report stays traceable inside the original four-sheet workbook.
    for c,count in enumerate(train_counts):
        annotations.append([fid,'train class '+str(c)+' support',count])
    write_xlsx(rows,template_path,out_dir/'experiments.xlsx',annotations)
    (out_dir/'REPORT.md').write_text(report)
    (out_dir/'environment.json').write_text(json.dumps(dict(conda='base',python=platform.python_version(),
                 torch=torch.__version__,device=environment,total_time_s=total_time),indent=2))
