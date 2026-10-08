//! 测试专用：把检测边界固化成 golden 快照，锁定端到端切分结果。
//!
//! 特征实现（fbank）改动后，特征值不再与旧的 Kaldi C++ 实现逐位相同，因此
//! 需要在**端到端切分边界**上建立回归基线：只要切分结果变化，测试就会失败
//! 并打印首个差异位置。
//!
//! 快照刷新方式见各调用点的文档；刷新后必须人工 review diff。

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// 拼出 golden 快照文件路径。
pub fn fixture_path(workspace_root: &Path, name: &str) -> PathBuf {
    workspace_root.join("tests/fixtures").join(name)
}

/// 把分节的字符串快照渲染成带注释头的文本。
///
/// `header` 是文件顶部的注释行（不含 `#` 前缀），`sections` 是
/// `(小节名, 每行内容)` 的有序列表。
pub fn render(header: &[String], sections: &[(String, Vec<String>)]) -> String {
    let mut out = String::new();
    for line in header {
        out.push_str("# ");
        out.push_str(line);
        out.push('\n');
    }
    for (name, lines) in sections {
        out.push('[');
        out.push_str(name);
        out.push_str("]\n");
        for line in lines {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// 解析分节的字符串快照，忽略空行与 `#` 注释。
///
/// 返回值顺序与文件中出现的小节顺序一致。
pub fn parse(text: &str) -> Result<Vec<(String, Vec<String>)>> {
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            sections.push((name.to_owned(), Vec::new()));
            continue;
        }
        let Some((_, lines)) = sections.last_mut() else {
            bail!("golden 快照在出现 [小节] 之前就有内容: {line:?}");
        };
        lines.push(line.to_owned());
    }
    if sections.is_empty() {
        bail!("golden 快照没有任何 [小节]");
    }
    Ok(sections)
}

/// 从解析结果中取出指定小节。
pub fn section<'a>(sections: &'a [(String, Vec<String>)], name: &str) -> Result<&'a [String]> {
    sections
        .iter()
        .find(|(section_name, _)| section_name == name)
        .map(|(_, lines)| lines.as_slice())
        .ok_or_else(|| anyhow::anyhow!("golden 快照缺少 [{name}] 小节"))
}

/// 把检测结果映射成 `<start_ms>-<end_ms>` 行，忽略自动生成的 span ID。
pub fn span_lines(spans: &[crate::TimeSpan]) -> Vec<String> {
    spans
        .iter()
        .map(|span| format!("{}-{}", span.range.start_ms, span.range.end_ms))
        .collect()
}

/// 按 golden 约定刷新快照；未设置 `UPDATE_GOLDEN` 时返回 `false`。
///
/// 这样测试既能在正常运行时做断言，也能通过 `UPDATE_GOLDEN=1` 一键重建基线。
pub fn write_if_requested(path: &Path, contents: String) -> Result<bool> {
    if std::env::var_os("UPDATE_GOLDEN").is_none() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, contents)?;
    eprintln!("已刷新 golden 快照: {}", path.display());
    Ok(true)
}

/// 读取 golden 快照，缺失时给出可执行的刷新提示。
pub fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|error| {
        anyhow::anyhow!(
            "缺少 golden 快照 {}（{error}）；执行 `UPDATE_GOLDEN=1 cargo test` 生成",
            path.display()
        )
    })
}

/// 逐行比对 span 列表，失败时打印首个差异位置以便定位。
pub fn assert_lines_eq(label: &str, actual: &[String], expected: &[String]) {
    if actual == expected {
        return;
    }
    let first_diff = actual
        .iter()
        .zip(expected)
        .position(|(lhs, rhs)| lhs != rhs);
    panic!(
        "{label} 与 golden 快照不一致：实际 {} 个 span，期望 {} 个，首个差异位于索引 {first_diff:?}\n实际: {actual:?}\n期望: {expected:?}",
        actual.len(),
        expected.len(),
    );
}
